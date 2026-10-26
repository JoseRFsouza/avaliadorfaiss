# Resultados — RAG FAISS

> Template para registrar cada rodada. Preencha **antes** de comparar com o
> projeto TF-IDF. Copie esta tabela para cada rodada (ou adicione seções).

## Configuração da rodada

| Item | Valor |
|---|---|
| Data | |
| Seed (`-s`) | |
| Questões (`-n`) | 5 sorteadas / 52 completas |
| Retrieval | FAISS IndexFlatIP (cosine), top_k=3 |
| Embeddings (retrieval) | nvidia/nemotron-3-embed-1b |
| LLM gerador | nvidia/nemotron-3.5-lightning-30b-a3b (temperature 0) |
| Juiz RAGAS | nvidia/nemotron-3.5-super-vl-120b-a12b |
| Embeddings (métrica) | nvidia/nemotron-3-embed-1b |
| Chunks | 26.039 (splitter 1000/200) |
| ragas | 0.4.3 |


## Rodada 2 — 22/10/2026: correcao das respostas vazias

| Item | Valor |
|---|---|
| Data | 22/10/2026 |
| Seed (`-s`) | n/a (`-n 52`, todas as questoes) |
| Questoes (`-n`) | 52 completas |
| Retrieval | FAISS IndexFlatIP (cosine), top_k=3 |
| Embeddings (retrieval) | nvidia/nemotron-3-embed-1b |
| LLM gerador | nvidia/nemotron-3.5-lightning-30b-a3b (temperature 0, **thinking ON, 16384 tokens**) |
| Juiz RAGAS | nvidia/nemotron-3.5-super-vl-120b-a12b |
| Embeddings (metrica) | nvidia/nemotron-3-embed-1b |
| Chunks | 26.039 (splitter 1000/200) |
| ragas | 0.4.3 |

### Diagnostico (comparacao com a rodada anterior, 4096 tokens)

| Configuracao | Vazias | Fallback | Tam. medio |
|---|---|---|---|
| Rodada 1 — 4096 tokens (original) | **7** | 19 | 464 chars |
| Teste — thinking OFF / 4096 | 0 | 20 | 398 chars |
| **Rodada 2 — thinking ON / 16384** | **0** | 23 | 566 chars |

**Causa raiz das 7 vazias:** `max_tokens` limita *raciocinio + resposta* juntos e o
raciocinio do modelo nao tem teto (medido: 2,3k-6,5k tokens/questao). Quando estourava
os 4096, a API devolvia `finish_reason=length` com `content` vazio. Nao era
deterministico (3 chamadas identicas: 2 respondiam, 1 vinha vazia).

**Por que thinking ON e nao OFF:** desligar o thinking tambem zera as vazias, mas fez
2 respostas boas virarem fallback (`gap-20-20`, `gt_014`) — o modo sem raciocinio nao
encontrava a informacao. Com 16384 tokens ambas voltaram a responder certo.

**Efeito liquido da Rodada 2 vs Rodada 1:** 7 vazias resolvidas, 3 fallbacks viraram
resposta real, 1 regressao genuina (`gt_110`), 2 ganhos de tamanho medio. As demais
diferencas sao variacao de execucao (temperature 0 ainda nao eh deterministico no NIM).

### Limitacao conhecida de recuperacao (nao eh bug)

Dos fallbacks restantes, a maioria eh **correta**: a informacao do ground truth nao
esta nos 3 contextos recuperados (cobertura de palavras do GT: 0-26%). Isso eh limite
do `top_k=3` denso; aumentar `top_k` ou rerank mudaria o metodo e quebraria a
comparacao justa com o projeto TF-IDF.

### Notas desta rodada

| Metrica | Rodada 1 (7 vazias) | Rodada 2 (0 vazias) |
|---|---|---|
| Faithfulness | 0.5080 | 0.1523 |
| Answer Relevancy | 0.3108 | 0.2715 |
| Context Precision | 0.6010 | 0.2949 |
| Context Recall | 0.5716 | 0.4956 |
| **Geral** | **0.4978** | **0.3036** |

> **ESTA TABELA ESTA ULTRAPASSADA -- ver "Rodada 3" abaixo.**
>
> A Rodada 2 foi corrigida: o juiz tinha a MESMA doenca do gerador (raciocinio
> estourando o orcamento de 8192 tokens). Em 23/10/2026 o `evaluate_ragas.py` passou a
> usar `JuizNVIDIA` com 16.384 tokens, raciocinio desligado no construtor e retry em
> resposta vazia. Os numeros da Rodada 2 (e da Rodada 1) continham notas 0 falsas, por
> falha do juiz -- nao por qualidade das
> respostas. Prova: `context_precision` nao depende da resposta, e diferia em 21/52
> questoes entre execucoes com recuperacao identica.
> **NAO compare essas notas ainda: o juiz esta falhando e os numeros nao sao confiaveis.**
> Duas evidencias: (1) durante a avaliacao ocorreu `429 Too Many Requests` da API NVIDIA;
> (2) `context_precision` **nao depende da resposta** (so de pergunta + contextos +
> ground truth) e a recuperacao eh identica nas duas rodadas -- mesmo assim ele difere em
> **21 das 52 questoes**. O juiz (`nemotron-3.5-super-vl-120b-a12b`) eh modelo de
> raciocinio com orcamento de 8192 tokens: mesma classe de falha do gerador. Quando
> estoura, devolve vazio, o RAGAS devolve NaN e o `fillna(0.0)` do `evaluate_ragas.py`
> transforma a falha em nota 0. Os dois CSVs tem muitos zeros (faithfulness: 20/52 e
> 42/52). Antes de concluir sobre qualidade, o juiz precisa do mesmo tratamento
> (orcamento maior + retry). So depois comparar as medias.


## Rodada 3 — 23/10/2026: juiz RAGAS corrigido

| Item | Valor |
|---|---|
| Data | 23/10/2026 |
| Questoes | 52 (mesmas da Rodada 2) |
| Gerador | igual a Rodada 2 (thinking ON, 16384 tokens) |
| Juiz | `nvidia/nemotron-3.5-super-vl-120b-a12b`, temperature 0, **raciocinio DESLIGADO**, 16384 tokens |
| Mudanca | `evaluate_ragas.py` passa a usar `JuizNVIDIA` (retry em resposta vazia) + `RunConfig` |

### Por que o juiz foi mexido

O juiz tinha a MESMA doenca do gerador. Medicao direta: num prompt simples ele gastou
**6.906 dos 8.192 tokens** disponiveis (84%), e o raciocinio variou de 1.176 a 6.895
tokens entre execucoes identicas. Quando estourava, devolvia vazio -> RAGAS devolvia
NaN -> o `fillna(0.0)` transformava a FALHA em nota 0.

Prova de que os numeros anteriores estavam corrompidos: `context_precision` **nao
depende da resposta** (so de pergunta + contextos + ground truth) e a recuperacao eh
identica, mas ele diferia em **21 das 52 questoes** entre duas execucoes.

Alem disso o `RunConfig` padrao do ragas (`timeout=180`, `max_workers=16`) matava jobs
lentos e provocava `429 Too Many Requests` por excesso de concorrencia.

### Notas (execucao concluida em 23/10/2026)

| Metrica | Rodada 1 | Rodada 3 (juiz corrigido) |
|---|---|---|
| Faithfulness | 0.4409 | 0.4409 |
| Answer Relevancy | 0.3108 | 0.0386 |
| Context Precision | 0.6010 | 0.3670 |
| Context Recall | 0.5716 | 0.4963 |
| **Geral** | **0.4978** | **0.3357** |

> **Esses numeros AINDA nao sao confiaveis, e agora sabemos exatamente por que.**
> A Rodada 3 rodou com o `fillna(0.0)` antigo (a correcao do relatorio veio depois),
> entao o CSV mistura **nota 0 genuina** com **falha do juiz** -- nao da para separar.
> Comparando so os valores nao-zero (que exclui tambem os zeros genuinos, sendo
> portanto um limite superior):

| Metrica | Rodada 1 (nao-zero) | Rodada 3 (nao-zero) | nao-zero R1 | nao-zero R3 |
|---|---|---|---|---|
| Faithfulness | 0.8256 | 0.8818 | 32/52 | 26/52 |
| Answer Relevancy | 0.5985 | 0.6688 | 27/52 | 3/52 |
| Context Precision | 0.8681 | 0.8674 | 36/52 | 22/52 |
| Context Recall | 0.9588 | 0.9558 | 31/52 | 27/52 |


### RESULTADO DEFINITIVO -- rodada com relatorio corrigido (23/10/2026)

| Metrica | Nota (validas) | Falhas do juiz | Cobertura |
|---|---|---|---|
| Faithfulness | 0.8131 | 28/52 | 24/52 (46%) |
| Answer Relevancy | 0.2216 | 46/52 | 6/52 (12%) |
| Context Precision | 0.4365 | 31/52 | 21/52 (40%) |
| Context Recall | 0.5603 | 23/52 | 29/52 (56%) |
| **Geral** | **0.6086** | -- | -- |

**Este eh o primeiro numero confiavel, porque agora a falha eh contada e nao**
**disfarcada de nota 0.** Leitura correta:

- **faithfulness 0,81** sobre 24 questoes: das que o juiz conseguiu avaliar, 81% das
  afirmacoes sao suportadas pelo contexto. Bom.
- **answer relevancy 0,22** sobre **6** questoes: amostra pequena demais para concluir
  anything -- nao use esse numero.
- **context precision 0,44** e **context recall 0,56**: ha espaco real de melhoria na
  recuperacao (o FAISS top-3 nao esta trazendo o contexto certo sempre).
- **geral 0,61**: media por questao das metricas validas.

**Risco desta rodada:** houve falha de DNS (`getaddrinfo failed`) e jobs de ~500 s perto
dos 86%, entao parte das 28 falhas de faithfulness pode ser infra, nao so parsing.
Rerodar em horario mais calmo (ou com `max_workers=2`) deve melhorar a cobertura.
**Conclusao honesta:** a qualidade REAL esta em torno de 0,83 / 0,60 / 0,87 / 0,96
(media dos valores nao-zero, consistente entre as duas rodadas). Os zeros eram falha
do juiz. A config nova melhorou faithfulness mas PIOROU a cobertura do answer relevancy
(3/52 validos) -- o raciocinio desligado nao serve bem para essa metrica.

**Proximo passo recomendado:** rerodar `python src\\evaluate_ragas.py`. O relatorio foi
corrigido: `fillna(0.0)` saiu, falhas ficam NaN, sao contadas por metrica e excluidas
das medias. Assim o proximo CSV separa nota 0 de falha, e da para escolher a config do
juiz (thinking on/off) com base em cobertura, nao em numero adulterado.

## Comparação head-to-head (mesma seed, mesmas questões)

| Projeto | Retrieval | Seed | Questões | Faithfulness | Relevancy | Precision | Recall | **Geral** |
|---|---|---|---|---|---|---|---|---|
| avaliadorRAGASTF-IDF | TF-IDF em camadas | | | | | | | |
| avaliadorRAGAS-FAISS (este) | FAISS top-k | | | | | | | |

## Observações

- (ex.: questões em que a recuperação acertou/errou, diferenças de comportamento
  entre os retrievers, custo/latência da indexação)

### Comparacao de configs do juiz (3 questões, 23/10/2026)

| Config | faithfulness validas | answer_relevancy validas | tempo/item |
|---|---|---|---|
| thinking OFF (atual) | 3/3 | 1/3 | ~50 s |
| thinking ON + 16384 | 2/3 | 1/3 | ~81 s |

**Nenhuma das duas resolve a confiabilidade.** Ambas deixam zeros/NaN, e o thinking ON
ainda eh ~60% mais lento. O juiz 
emotron-3.5-super-vl-120b-a12b eh instavel com os
prompts do RAGAS nesta conta (429 + parsing). Por isso a config atual ficou com o
thinking DESLIGADO (mais rapido, cobertura equivalente).

O que da para afirmar com seguranca: a qualidade REAL esta na faixa dos valores
nao-zero (~0,83 faithfulness, ~0,87 context precision, ~0,96 context recall),
consistente entre rodadas. As medias impressas pelo script NAO sao confiaveis enquanto
houver falha misturada. Para um numero definitivo, rerodar com o relatorio corrigido
(que agora conta falhas por metrica) ou trocar o juiz por um modelo sem raciocinio.
### Conclusao da investigacao do juiz (23/10/2026)

Foram testadas 4 configuracoes do juiz, todas nas mesmas 3 questoes:

| Config | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| original (8192, thinking ON) | 20/52 validos* | 25/52* | 16/52* | 21/52* |
| thinking OFF, 4 workers | 3/3 | 1/3 | 1/3 | 2/3 |
| thinking ON, 16384 | 2/3 | 1/3 | 2/3 | 2/3 |
| thinking OFF, 2 workers | 3/3 | 1/3 | 1/3 | 2/3 |

\* numeros da rodada completa de 52, para referencia de escala.

**Conclusao:** a concorrencia (max_workers 2 vs 4) nao muda nada, e pensar ou nao tambem
nao. O juiz 
emotron-3.5-super-vl-120b-a12b eh instavel com os prompts do RAGAS nesta
conta -- mistura 429 da API com saida que o RAGAS nao consegue parsear. Nao ha
configuracao de cliente que resolva isso.

**O que foi entregue de concreto:** o relatorio nao mente mais. illna(0.0) removido,
falhas contadas por metrica, medias sobre validas. Com isso, da para medir a cobertura
real e decidir com dados.

**Se quiser numero confiavel:** trocar o juiz por um modelo sem raciocinio (ex.: um
llama-3.3-70b ou similar do catalogo NVIDIA) — ai a avaliacao fica rapida e estavel.
Isso muda o juiz, entao seria uma rodada nova, nao comparavel com as anteriores.
### Tentativa de juiz sem raciocinio (23/10/2026)

i21labs/jamba-1.5-large-instruct aparece no catalogo (/v1/models) mas devolve
**404 Not Found** ("Function not found for account") ao ser chamado -- nao esta de fato
disponivel. meta/llama-3.3-70b-instruct e meta/llama-3.1-70b-instruct tambem nao
estao no catalogo.

**Resultado:** nao existe hoje, nesta conta, um juiz sem raciocinio de qualidade para
substituir o nemotron. A avaliacao RAGAS fica limitada pela instabilidade do juiz atual.

**Entregavel concreto desta investigacao:** o relatorio nao mente mais (illna(0.0)
removido, falhas contadas por metrica, medias sobre validas). Com isso, qualquer rodada
futura mostra a cobertura real -- da para saber se o numero pode ser usado ou nao.
### Embeddings de metrica (23/10/2026) -- descartado

NVIDIAEmbeddings(nvidia/nemotron-3-embed-1b) funciona normalmente (mbed_query e
mbed_documents retornam 2048 dimensoes). O UserWarning "type is unknown and inference
may fail" eh benigno e NAO explica as falhas do answer relevancy -- a causa eh o juiz.
