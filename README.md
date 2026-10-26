# Benchmark RAG-FAISS — Boeing 737 AMM + Avaliação RAGAS

Projeto **irmão** do [`avaliadorRAGASTF-IDF`](../avaliadoresRAGAS): mesma base de
conhecimento, mesmas questões, mesmo avaliador — **só a recuperação muda**. Aqui o
retrieval é **denso (embeddings + FAISS, busca vetorial global top-k)**; lá é
TF-IDF em camadas (TOC → arquivo ATA → task). O objetivo é comparar os dois de
forma justa.

## O que é igual / o que muda (regra de ouro da comparação)

| Elemento | Projeto TF-IDF | Este projeto (FAISS) |
|---|---|---|
| Corpus, chunking (1000/200), 26.039 chunks | ✅ igual | ✅ igual |
| Questões (52) e ground truth | ✅ igual | ✅ igual |
| **Avaliação (`evaluate_ragas.py`)** | ✅ **cópia idêntica** | ✅ **cópia idêntica** |
| Juiz RAGAS + embeddings de métrica | `nemotron-3.5-super-vl-120b-a12b` / `nemotron-3-embed-1b` | igual |
| LLM gerador, temperatura, prompt | `nemotron-3.5-lightning-30b-a3b`, temp 0 | igual |
| Contextos por questão (`top_k=3`) | ✅ igual | ✅ igual |
| **Retrieval** | TF-IDF em camadas | **FAISS top-k global** ← única diferença |
| Uso de ground truth na recuperação | nunca | nunca |

> **Por que isso importa:** se mudar mais de uma coisa entre os projetos, não se
> sabe o que explicou a diferença de nota. Aqui, a única variável é o método de
> recuperação.

## Estrutura

```
avaliadorRAGAS-FAISS/
├── .gitignore                    # protecoes (nao versione .env, venvs, data/, GT)
├── README.md                     # este arquivo
├── results.md                    # template para registrar resultados
├── requirements.txt              # dependencias com versoes fixadas
├── .env                          # SUA chave da API (nunca sobe pro git)
├── ground_truth_approved.json    # 52 questoes curadas (local, nao versionado)
├── data/
│   ├── train.json                # gerado pelo prepare_dataset.py
│   └── corpus/                   # 48 JSONs do manual + amm_toc_database.json (local)
├── output/                       # predictions.json, cache de embeddings, CSV
└── src/
    ├── rag_faiss.py              # NOVO: pipeline FAISS (retrieval denso)
    ├── evaluate_ragas.py         # copia IDENTICA do projeto TF-IDF (o juiz)
    └── prepare_dataset.py        # copia identica (converte GT -> train.json)
```

## Como executar — passo a passo completo

Abra o PowerShell **na pasta do projeto** e siga. Nada de decorar comando: é só
copiar daqui.

### 1. Criar o ambiente virtual (uma vez só)

```powershell
cd C:\Users\joser\Desktop\cursos\avaliadorRAGAS-FAISS
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

⚠️ **Crie o venv UMA única vez e com o nome `.venv`.** Não crie `venv/` (sem
ponto) nem `activate/` — ter vários venvs é a principal fonte de confusão
("ModuleNotFoundError" do nada). Confirma que deu certo:

```powershell
python -c "import sys; print(sys.prefix)"
# deve terminar com \avaliadorRAGAS-FAISS\.venv
```

### 2. Instalar dependências

```powershell
pip install -r requirements.txt
# se a instalacao do ragas falhar (scikit-network sem wheel para Python 3.14):
pip install --no-deps ragas==0.4.3
```

> Este projeto **não precisa de torch/sentence-transformers** — o FAISS é binário
> (`faiss-cpu`), então não há o problema de DLL do Visual C++.

### 3. Chave da API

O arquivo `.env` já está copiado neste projeto. Se um dia precisar trocar a chave,
edite o `.env` **direto no VS Code** (não cole pelo chat):

```
NVIDIA_API_KEY=nvapi-...
```

Obtenha em <https://build.nvidia.com> → Login/Get API Key.

### 4. Rodar o pipeline RAG (FAISS)

```powershell
python src\rag_faiss.py -r -n 5
```

- `-r` = sorteia 5 questões aleatórias (seed nova a cada execução)
- `-s 123` = seed fixa (use a **mesma** nos dois projetos na comparação)
- `-n 52` = todas as questões (recomendado para o resultado "oficial")
- `--max-chunks 500` = modo teste rápido (índice parcial, leva segundos)

**A primeira execução leva vários minutos**: ela embedda todos os 26.039 chunks na
API da NVIDIA (em lotes de 100) e salva o cache em `output/faiss_embeddings.npy`
+ `output/faiss_chunks.json`. As execuções seguintes reutilizam o cache
(instantâneo). Se mudar o corpus ou o modelo de embeddings, o cache é refeito
sozinho (há um fingerprint).

Saída: `output/predictions.json` (mesmo schema do projeto TF-IDF).

> **Lentidão e timeouts:** o gerador é um modelo de *raciocínio* — cada questão
> pode levar de alguns segundos a alguns minutos. O `rag_faiss.py` usa
> `timeout=300` e até 4 tentativas com backoff, e grava o `predictions.json`
> **a cada questão**: se a rede cair no meio, o que já foi respondido fica salvo
> (basta rodar de novo para completar).

### 5. Avaliar com RAGAS (juiz independente)

```powershell
python src\evaluate_ragas.py
```

Rodas as 4 métricas (faithfulness, answer relevancy, context precision, context
recall), imprime o relatório por questão e salva
`output/resultado_ragas_metrics.csv`.

### 6. (Opcional) Regenerar o train.json

Só necessário se trocar o `ground_truth_approved.json`:

```powershell
python src\prepare_dataset.py
```

## Como funciona por dentro

```
Pergunta ──► embedda a pergunta (nemotron-3-embed-1b, input_type=query)
                 │  cosine similarity (vetores normalizados)
                 ▼
        FAISS IndexFlatIP sobre 26.039 chunks embeddados ──► top-3 contextos
                 │
                 ▼
        LLM gerador (nemotron-3.5-lightning-30b-a3b, temp 0, thinking ON, 16384 tokens) + MESMO prompt
                 │
                 ▼
        predictions.json ──► evaluate_ragas.py (RAGAS) ──► CSV + notas
```

- Índice: `IndexFlatIP` (busca exata; 26k chunks não precisa de índice aproximado)
- Similaridade: cosseno (produto interno com vetores L2-normalizados)
- Cache: embeddings dos chunks ficam em `output/` — apagar força re-embutir
- **Gerador: `thinking` LIGADO com orcamento de 16.384 tokens** (`GERADOR_MAX_TOKENS`).
  O modelo eh de raciocinio e o `max_tokens` limita *raciocinio + resposta* juntos. O
  raciocinio nao tem teto (medido: 2,3k-6,5k tokens por questao), entao com 4096 ele
  estourava o orcamento e a API devolvia `finish_reason=length` com `content`
  **VAZIO** -- foi a causa das **7 respostas vazias em 52** no run de 22/10/2026.
  Duas opcoes foram testadas: desligar o thinking zera as vazias, mas fez **2
  respostas boas virarem fallback** (o modo sem raciocinio nao achava a informacao);
  com thinking ligado e 16.384 tokens essas 2 voltaram a responder certo. Por isso o
  thinking fica ligado e o orcamento subiu. `_invocar_llm()` ainda trata resposta
  vazia como falha e tenta de novo, com marcador explicito se persistir.

## Metodologia de comparação justa (leia antes de comparar)
0. **Mesmo gerador, mesma configuração:** os modelos nemotron são de
   *raciocínio* e, sem `max_tokens` explícito, devolvem `content` **vazia** (a
   reasoning consome todo o orçamento). O `rag_faiss.py` já usa
   `max_completion_tokens=16384` (thinking LIGADO) — **aplique o mesmo valor no projeto
   TF-IDF**, junto com o
   mesmo modelo de gerador. Nota de outubro/2026: o
   `nemotron-3-super-120b-a12b` saiu do catálogo da conta (404); os resultados
   antigos do TF-IDF usavam esse modelo, então regravações novas devem usar o
   mesmo gerador novo nos dois projetos para serem comparáveis entre si.
1. **Mesma seed:** rode os dois projetos com `-s <mesmo número>` (ex.: `-s 42`)
   na mesma quantidade de questões. Sem seed fixa, cada run sorteia questões
   diferentes e as notas não são comparáveis.
2. **Prefira as 52 questões:** com 5 questões a nota oscila muito (de 0,35 a 0,58
   entre sorteios). Para o resultado "oficial", use `-n 52` nos dois.
3. **Um variável só:** não mude gerador, juiz, prompt, top_k, chunking ou o
   avaliador entre os projetos. O que pode mudar é só o retrieval.
4. **Configure antes, não depois:** escolha o modelo de embeddings e o top_k
   **antes** de ver resultados — tunar até "ganhar" invalida o experimento.
5. **Registre tudo** no `results.md` (template pronto): seed, data, modelos,
   versões, nota agregada e por questão.
6. **Múltiplas seeds:** o mais rigoroso é rodar 3–5 seeds em cada projeto e
   comparar as médias.

## Versionamento (opcional)

```powershell
git init
git add .
git commit -m "Benchmark RAG-FAISS + avaliacao RAGAS"
git remote add origin https://github.com/SEU_USUARIO/NOME_REPO.git
git push -u origin main
```

O `.gitignore` já protege: `.env`, `.venv/`, `venv/`, `activate/`, `data/`,
`ground_truth_approved.json` e `output/`. **Nunca** commite a chave da API nem o
material do manual da Boeing (proprietário).

## Problemas conhecidos (e a solução)

| Sintoma | Causa | Solução |
|---|---|---|
| `ModuleNotFoundError: No module named 'dotenv'` | venv errado (ex.: `venv` sem ponto, ou venv recriado) | use **só** o `.venv`; confirme com `python -c "import sys; print(sys.prefix)"` |
| `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` | `langchain-community` 0.4.2 instalado | `pip install langchain-community==0.4.1` |
| `Invalid JWT serialization` / 401 | variável de ambiente `NVIDIA_API_KEY=sua_chave_aqui` sombreando o `.env` | os scripts usam `load_dotenv(override=True)`; se persistir, apague a variável do Windows |
| 404 "Function not found for account" | modelo não disponível na sua conta | use os modelos deste README |
| ragas não instala (scikit-network) | sem wheel para Python 3.14 | `pip install --no-deps ragas==0.4.3` |
| 404 no LLM gerador | modelo saiu do catálogo da conta (ex.: `nemotron-3-super` em 20/10/2026) | ajuste `GERADOR_MODEL` no `rag_faiss.py` (e no projeto TF-IDF, o mesmo modelo) |
| `requests.exceptions.ReadTimeout` (`integrate.api.nvidia.com`, read timeout=60) | o `ChatNVIDIA` tem timeout de leitura de **60 s** por padrão; os modelos nemotron são de raciocínio e, com 3 chunks de contexto, passam disso | o `rag_faiss.py` já usa `timeout=300` (`LLM_TIMEOUT`) e retry com backoff; o `predictions.json` é salvo a cada questão, então um timeout não descarta o run |
| Respostas **vazias** (`answer: ""`) | o `max_tokens` limita *raciocínio + resposta* juntos; o raciocínio do modelo estoura o orçamento e a API devolve `finish_reason=length` sem conteúdo (eram 7 em 52) | `GERADOR_MAX_TOKENS = 16384` (thinking LIGADO) + `_invocar_llm()` tratando resposta vazia como falha; desligar o thinking também zera as vazias, mas fez 2 respostas boas virarem fallback |
| Notas RAGAS muito baixas / muitos 0,0 | o **juiz** (`nemotron-3.5-super-vl-120b-a12b`) tambem eh de raciocinio e, com 8192 tokens, estourava o orcamento (medido: 6.906 de 8.192 tokens num prompt simples) -> vazio -> NaN -> `fillna(0.0)` virava nota 0; tambem houve `429` e `TimeoutError` (RunConfig padrao: timeout=180, 16 workers) | `evaluate_ragas.py` agora usa `JuizNVIDIA` (retry em resposta vazia), `JUIZ_MAX_TOKENS=16384`, raciocinio desligado via `chat_template_kwargs` no construtor e `RunConfig(timeout=900, max_workers=4)`; nao muda metricas, prompt, modelo nem seed |
| Notas RAGAS misturando falha com nota 0 | o `fillna(0.0)` transformava a falha do juiz (NaN) em nota 0, corrompendo as médias | removido: falhas ficam NaN, são **contadas por métrica** e excluídas das médias; o resumo final mostra quantas questões falharam |
| Answer Relevancy penaliza respostas corretas | o RAGAS 0.4.3 compara a pergunta em modo `query` com perguntas geradas em modo `passage`; esses modos usam espaços vetoriais distintos no `nvidia/nemotron-3-embed-1b` | `src/metricas_ragas.py` codifica ambos no modo `query` e faz o embedding em lote; aplicado igualmente nos projetos FAISS, TF-IDF, MISTURA e RRF. HTTP 429 do juiz continuam sendo tratados separadamente por retry/execução serial |
