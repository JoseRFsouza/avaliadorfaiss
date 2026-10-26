"""Pipeline RAG com recuperacao DENSA (FAISS) sobre o manual AMM Boeing 737.

Projeto irmao do avaliadorRAGASTF-IDF (TF-IDF). A UNICA diferenca de metodo eh o
retrieval: aqui usa embeddings + FAISS (busca vetorial global top-k) em vez de
TF-IDF em camadas (TOC -> arquivo ATA -> task).

Todo o resto eh identico de proposito (comparacao justa):
  - mesmo corpus, mesmo chunking (1000/200) -> mesmos 26.039 chunks
  - mesmo LLM gerador, mesma temperatura (0), mesmo prompt
  - mesmo schema de predictions.json -> mesmo evaluate_ragas.py (copia identica)
  - mesmo top_k (3 contextos), mesmo sorteio de questoes (mesma logica de seed)


ATENCAO (comparabilidade): em 22/10/2026 o orcamento do gerador subiu de 4096 para
16384 tokens (GERADOR_MAX_TOKENS). Com 4096, 7 de 52 respostas vinham VAZIAS porque
o raciocinio do modelo estourava o max_tokens (finish_reason=length, content vazio).
O thinking continua LIGADO -- desligar eliminava as vazias mas fazia 2 respostas boas
virarem fallback. Se for comparar com resultados antigos (ou com o projeto TF-IDF),
aplique a mesma configuracao nos dois projetos -- senao a comparacao fica injusta.

Uso:
    python src/rag_faiss.py -n 5            # 5 primeiras questoes
    python src/rag_faiss.py -r -n 5         # 5 questoes sorteadas (seed aleatoria)
    python src/rag_faiss.py -r -n 5 -s 123  # seed fixa (comparacao head-to-head)
    python src/rag_faiss.py -r -n 5 --max-chunks 2000   # teste rapido (subset)

A primeira execucao embedda todos os chunks (leva varios minutos) e salva o cache
em output/faiss_*. As execucoes seguintes reutilizam o cache.
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)  # .env do projeto prevalece sobre variaveis de ambiente

# ------------------------------------------------------------------ caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "data" / "corpus"
TRAIN_FILE = BASE_DIR / "data" / "train.json"
OUTPUT_DIR = BASE_DIR / "output"
PRED_FILE = OUTPUT_DIR / "predictions.json"
CACHE_VECTORS = OUTPUT_DIR / "faiss_embeddings.npy"
CACHE_CHUNKS = OUTPUT_DIR / "faiss_chunks.json"
TOC_FILE = CORPUS_DIR / "amm_toc_database.json"

# ------------------------------------------------------- config (fixa, para comparacao justa)
GERADOR_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"  # nemotron-3-super saiu do catalogo da conta (404)
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"        # retrieval (diferenca do metodo)
CHUNK_SIZE = 1000                                      # mesmo splitter do TF-IDF
CHUNK_OVERLAP = 200
TOP_K = 3                                              # mesmo orcamento de contexto
BATCH_SIZE = 100
NVIDIA_EMB_URL = "https://integrate.api.nvidia.com/v1/embeddings"

# O ChatNVIDIA tem timeout de LEITURA de 60s por padrao. Os modelos nemotron sao
# de raciocinio: com 3 chunks de contexto (~3000 chars) e max_tokens=4096 a
# geracao passa de 60s e estoura em requests.ReadTimeout. 300s + retry abaixo.
LLM_TIMEOUT = 300
GERADOR_TENTATIVAS = 4

# O gerador eh modelo de RACIOCINIO e o max_tokens limita (raciocinio + resposta)
# juntos. O raciocinio nao tem teto (medido: 2.3k-6.5k tokens/questao), entao com
# 4096 ele estourava o orcamento e a API devolvia finish_reason=length com content
# VAZIO -- foi a causa das 7 respostas vazias em 52 (9, 25, 34, 40, 45, 50, 51).
#
# Duas configuracoes foram testadas em 22/10/2026:
#   thinking OFF: 0 vazias, MAS 2 respostas boas viraram fallback (o modo sem
#                 raciocinio nao encontrava a informacao que o raciocinio achava).
#   thinking ON + 16384: as 2 questoes regredidas voltaram a responder certo.
# Por isso o thinking fica LIGADO e o orcamento subiu para 16384 (folga grande
# sobre os 6.5k observados). _invocar_llm() ainda trata resposta vazia como falha.
GERADOR_THINKING = True
GERADOR_MAX_TOKENS = 16384


# ------------------------------------------------------------------ corpus (loader identico ao TF-IDF)
def carregar_documentos():
    """Carrega as secoes do manual (JSONs pre-extraidos) como Documents."""
    docs = []
    for json_path in sorted(CORPUS_DIR.glob("*.json")):
        if json_path.name == TOC_FILE.name:
            continue  # amm_toc_database.json nao eh conteudo de task
        with open(json_path, "r", encoding="utf-8") as f:
            secoes = json.load(f)
        if not isinstance(secoes, list):
            continue
        for item in secoes:
            conteudo = item.get("content", "")
            if not conteudo:
                continue
            docs.append(Document(
                page_content=f"{item.get('title', '')}\n{conteudo}",
                metadata={"source": json_path.name, "task_id": item.get("task_id", "")},
            ))
    if not docs:
        raise FileNotFoundError(
            f"Nenhuma secao encontrada em {CORPUS_DIR}. "
            "Obtenha os JSONs do manual AMM e coloque-os em data/corpus/."
        )
    return docs


def gerar_chunks(docs):
    """Mesmo RecursiveCharacterTextSplitter do projeto TF-IDF -> mesmos chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    return [c.page_content for c in chunks]


# ------------------------------------------------------------------ embeddings (API NVIDIA, com cache)
def _fingerprint():
    """Identifica a configuracao do indice: se mudar, o cache eh descartado."""
    h = hashlib.sha256()
    h.update(f"{CHUNK_SIZE}|{CHUNK_OVERLAP}|{EMBEDDING_MODEL}".encode())
    for p in sorted(CORPUS_DIR.glob("*.json")):
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]


def _embed_lote(textos, input_type="passage", tentativas=4):
    """Embuta um lote de textos na API NVIDIA, com retry/backoff."""
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY nao encontrada. Configure o arquivo .env.")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": EMBEDDING_MODEL, "input": textos, "input_type": input_type}
    delay = 3
    for tentativa in range(1, tentativas + 1):
        try:
            resp = requests.post(NVIDIA_EMB_URL, headers=headers, json=payload, timeout=180)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            dados = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
            return np.array([d["embedding"] for d in dados], dtype="float32")
        except Exception as e:
            if tentativa == tentativas:
                raise
            print(f"   falha no lote ({e}); tentativa {tentativa + 1}/{tentativas} em {delay}s...")
            time.sleep(delay)
            delay *= 2


def _invocar_llm(llm, prompt, tentativas=GERADOR_TENTATIVAS):
    """Chama o LLM gerador com retry/backoff.

    Alem de timeout/erro de rede, trata RESPOSTA VAZIA como falha: com o thinking
    ligado o raciocinio pode consumir todo o orcamento de tokens e a API devolve
    finish_reason=length sem conteudo. Se persistir, devolve um marcador explicito
    em vez de derrubar a execucao inteira (o run de 52 questoes eh longo).
    """
    delay = 5
    for tentativa in range(1, tentativas + 1):
        try:
            conteudo = llm.invoke(prompt).content
            if conteudo and conteudo.strip():
                return conteudo
            falha = "resposta vazia (raciocinio estourou o orcamento de tokens)"
        except Exception as e:
            falha = f"{type(e).__name__}: {e}"
        if tentativa == tentativas:
            break
        print(f"   falha na geracao ({falha}); tentativa {tentativa + 1}/{tentativas} em {delay}s...")
        time.sleep(delay)
        delay *= 2
    marcador = f"[GERACAO FALHOU: {falha} apos {tentativas} tentativas]"
    print(f"   ⚠️  {marcador}")
    return marcador


def _carregar_cache(fp):
    """Retorna (vetores, chunks) do cache se valido para o fingerprint, senao ([], [])."""
    if not (CACHE_VECTORS.exists() and CACHE_CHUNKS.exists()):
        return np.empty((0, 0), dtype="float32"), []
    try:
        meta = json.loads(CACHE_CHUNKS.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fp:
            return np.empty((0, 0), dtype="float32"), []
        vetores = np.load(CACHE_VECTORS)
        return vetores, meta.get("chunks", [])
    except Exception:
        return np.empty((0, 0), dtype="float32"), []


def _salvar_cache(fp, vetores, chunks):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_VECTORS, vetores)
    CACHE_CHUNKS.write_text(
        json.dumps({"fingerprint": fp, "count": len(chunks), "chunks": chunks},
                   ensure_ascii=False), encoding="utf-8")


def construir_indice(chunk_texts):
    """Embuta os chunks (com cache/retomada) e devolve o indice FAISS (cosine)."""

    fp = _fingerprint()
    vetores, chunks_cache = _carregar_cache(fp)

    if len(chunks_cache) == len(chunk_texts):
        print(f"   cache de embeddings reutilizado ({len(chunks_cache)} chunks).")
    else:
        if chunks_cache:
            print(f"   retomando do cache: {len(chunks_cache)}/{len(chunk_texts)} chunks embeddados.")
        else:
            print(f"   embeddando {len(chunk_texts)} chunks (lotes de {BATCH_SIZE})... "
                  "a primeira execucao leva varios minutos.")
        pendentes = chunk_texts[len(chunks_cache):]
        partes = [vetores] if chunks_cache else []
        total = len(chunks_cache)
        for inicio in range(0, len(pendentes), BATCH_SIZE):
            lote = pendentes[inicio:inicio + BATCH_SIZE]
            v = _embed_lote(lote, input_type="passage")
            partes.append(v)
            total += len(lote)
            if (inicio // BATCH_SIZE) % 10 == 9:  # salva progresso a cada 10 lotes
                _salvar_cache(fp, np.vstack(partes), chunk_texts[:total])
                print(f"   ...{total}/{len(chunk_texts)} embeddados")
        vetores = np.vstack(partes)
        _salvar_cache(fp, vetores, chunk_texts)
        print(f"   embeddings concluidos: {vetores.shape}.")

    # cosine similarity = produto interno com vetores normalizados
    vetores = np.ascontiguousarray(vetores, dtype="float32")
    faiss.normalize_L2(vetores)
    indice = faiss.IndexFlatIP(vetores.shape[1])
    indice.add(vetores)
    return indice


def recuperar_contextos(pergunta, indice, chunk_texts, top_k=TOP_K):
    """Recuperacao densa global: embedda a pergunta e devolve os top_k chunks."""
    q = _embed_lote([pergunta], input_type="query")
    faiss.normalize_L2(q)
    _, idxs = indice.search(q, top_k)
    return [chunk_texts[i] for i in idxs[0] if 0 <= i < len(chunk_texts)]


# ------------------------------------------------------------------ pipeline
PROMPT_TEMPLATE = PromptTemplate.from_template("""
Você é um especialista técnico de manutenção aeronáutica do Boeing 737 (AMM).
Com base estritamente no contexto recuperado dos manuais abaixo, responda à pergunta do mecânico de forma precisa e direta. Se a informação não estiver contida no contexto, declare que não possui informações suficientes no manual fornecido.

Contexto Técnico Recuperado:
{context}

Pergunta: {question}

**IMPORTANTE: Responda SEMPRE em português brasileiro técnico.**`r`n`r`nResposta Técnica:
""")


def executar_pipeline_rag(limit=None, is_random=False, seed=None, max_chunks=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Carregando seções do manual Boeing 737 em data/corpus/ (JSONs pré-extraídos)...")
    docs = carregar_documentos()
    print(f"   {len(docs)} seções carregadas.")

    print("2. Gerando chunks (splitter 1000/200 — igual ao projeto TF-IDF)...")
    chunk_texts = gerar_chunks(docs)
    if max_chunks:
        chunk_texts = chunk_texts[:max_chunks]
        print(f"   MODO TESTE: usando apenas os primeiros {len(chunk_texts)} chunks.")
    print(f"   {len(chunk_texts)} chunks.")

    print(f"3. Construindo índice FAISS (embeddings {EMBEDDING_MODEL})...")
    indice = construir_indice(chunk_texts)

    print("4. Inicializando o LLM gerador...")
    # max_completion_tokens explicito: os modelos nemotron sao de raciocinio e
    # consomem o orcamento padrao inteiro, devolvendo content vazia sem isto
    # timeout explicito: o padrao do ChatNVIDIA (60s de leitura) estoura em
    # requests.ReadTimeout justamente nesses modelos de raciocinio
    llm = ChatNVIDIA(
        model=GERADOR_MODEL,
        temperature=0.0,
        max_completion_tokens=GERADOR_MAX_TOKENS,
        timeout=LLM_TIMEOUT,
    )
    # thinking: o gerador eh modelo de raciocinio. Deixar LIGADO preserva a
    # qualidade (2 questoes em que o modo sem raciocinio caia no fallback); o
    # orcamento de 16384 tokens e o suficiente para o raciocinio nao estourar e
    # devolver content vazio. Para testar o modo sem raciocinio, use False.
    if not GERADOR_THINKING:
        llm = llm.with_thinking_mode(False)
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        perguntas_gt = json.load(f)

    if is_random and limit:
        if seed is None:
            seed = random.randrange(1, 1_000_000)  # sorteio realmente variável
        random.seed(seed)
        qtd = min(limit, len(perguntas_gt))
        perguntas_gt = random.sample(perguntas_gt, qtd)
        print(f"🎲 Selecionadas {qtd} questões aleatórias (Seed {seed}).")
    elif limit:
        perguntas_gt = perguntas_gt[:limit]
        print(f"⚡ Selecionadas as primeiras {len(perguntas_gt)} questões.")

    predictions = []
    print("\n" + "=" * 80)
    print(" EXECUTANDO GERADOR RAG (FAISS + LLM)")
    print("=" * 80)

    for i, item in enumerate(perguntas_gt, 1):
        pergunta = item.get("question")
        gabarito = item.get("ground_truth")

        contextos = recuperar_contextos(pergunta, indice, chunk_texts, top_k=TOP_K)
        contexto_unificado = "\n\n--- CHUNK ---\n\n".join(contextos)
        prompt_formatado = PROMPT_TEMPLATE.format(context=contexto_unificado, question=pergunta)
        resposta_llm = _invocar_llm(llm, prompt_formatado)

        print(f"\n[Questão {i}/{len(perguntas_gt)}]")
        print(f"❓ PERGUNTA: {pergunta}")
        print(f"📘 GROUND TRUTH: {gabarito}")
        print(f"🤖 RESPOSTA GERADA: {resposta_llm}")
        print("-" * 80)

        predictions.append({
            "question_id": item.get("question_id"),
            "question": pergunta,
            "answer": resposta_llm,
            "contexts": contextos,
            "ground_truth": gabarito,
        })

        # salva a cada questao: um timeout/erro de rede nao descarta a execucao
        with open(PRED_FILE, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

    with open(PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Execução concluída. Respostas salvas em: {PRED_FILE}")
    print("   Próximo passo: python src/evaluate_ragas.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline RAG com FAISS + avaliação RAGAS")
    parser.add_argument("-n", "--limit", type=int, default=None, help="Quantidade de perguntas")
    parser.add_argument("-r", "--random", action="store_true", help="Amostragem aleatória")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Semente do sorteio (padrão: aleatória)")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="MODO TESTE: usa apenas os N primeiros chunks do índice")

    args = parser.parse_args()
    executar_pipeline_rag(
        limit=args.limit,
        is_random=args.random,
        seed=args.seed,
        max_chunks=args.max_chunks,
    )
