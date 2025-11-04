DISTO Reproduction (SepT + DistilRoBERTa)

Resumo

- Este projeto reproduz, de modo funcional e leve, a métrica DISTO conforme o paper “DISTO: Textual Distractors for Multiple Choice Reading Comprehension Questions using Negative Sampling” (EDM 2024). Implementamos o modelo SepT com DistilRoBERTa para pontuar a qualidade de distractors em questões de múltipla escolha (MCQ). O pipeline inclui: pré-processamento (RACE), negative sampling (4 métodos), treino e avaliação.

Referência (paper):

- DISTO: Textual Distractors for Multiple Choice Reading Comprehension Questions using Negative Sampling (EDM 2024).

Ambiente

- Requer Python 3.10. CUDA é opcional; CPU funciona, mas mais lento.

Comandos iniciais

- Crie e ative o ambiente virtual, instale dependências e baixe o modelo spaCy:

```
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Pipeline

1) Pré-processamento (RACE → Parquet flatten com bons distractors label=1.0)

```
python preprocess/prepare_dataset.py --config config/datasets.yaml --out data/processed
```

2) Negative Sampling (gera dataset com bons + negativos, 4 métodos)

```
python sampling/negative_sampling.py --in data/processed --out data/ns --k 200
```

3) Treino (SepT + DistilRoBERTa, MSE, AdamW lr=1e-5, early stopping simples)

```
python train_disto.py --data data/ns --model distilroberta-base --output runs/disto-sept --epochs 2 --bsz 8
```

4) Avaliação (MAE e correlação de Pearson no test)

```
python evaluate_disto.py --ckpt runs/disto-sept/best.ckpt --data data/ns
```

Arquitetura e arquivos

- `config/datasets.yaml`: configuração dos datasets. Inicialmente inclui RACE (config “all”) com splits. Há placeholders comentados para CosmosQA, DREAM, MCTest, MCScript, Quail, SCIQ com nota de que cada um precisa de um mapper para (article, question, answer, distractors[]).
- `preprocess/prepare_dataset.py`: carrega RACE (via `datasets.load_dataset("race", "all")`), mapeia para linhas `article, question, answer, distractor, label=1.0` (flatten), valida e salva Parquet por split.
- `sampling/clustering_features.py`: extrai features POS TF, NER TF e length com spaCy e fornece `fuse_features` para concatenação com o embedding CLS de DistilRoBERTa.
- `sampling/negative_sampling.py`: gera negativos (4 métodos): Answer Replication, Random (pool global do treino), Farthest in Cluster (KMeans k=200 com features = CLS + POS TF + NER TF + length) e BERT [MASK] filling (troca uma palavra por previsão de MLM). Salva Parquets com bons + negativos.
- `models/model_sept.py`: implementação do SepT (DistilRoBERTa) que concatena `[QUES] {Q} [ANS] {A} [DIS] {D} [ART] {Ar}` e aplica uma head linear no CLS com sigmoid.
- `train_disto.py`: treino do SepT com MSE, AdamW e scheduler linear com 10% warmup. Early stopping pelo menor MSE de validação, salvando `runs/disto-sept/best.ckpt`.
- `evaluate_disto.py`: carrega o checkpoint e avalia no test: MAE (%) e Pearson r (com p-value).

Execução rápida e amostragem

- Todos os scripts aceitam amostragem para rodar rapidamente em CPU, por exemplo: `--sample_train 20000`, `--sample_validation 5000`, `--sample_test 5000` (quando aplicável). Também há `--max_length` nos embeddings do negative sampling para reduzir custo (padrão 128/256) e `--batch_size` para controlar uso de memória.

Compatibilidade

- Python 3.10; CUDA opcional; dependências fixadas em versões estáveis no `requirements.txt`.

Limitações e observações

- O KMeans com k=200 em pools grandes pode ser custoso em CPU/memória. Ajuste `--k`, `--max_length` e `--sample_*` conforme recursos.
- O método BERT [MASK] filling é uma aproximação funcional (substitui 1 token por previsão top-k do `bert-base-uncased`, garantindo troca). Em casos sem tokens elegíveis ou sem candidatos válidos, faz fallback para Random negativo.
- Para outros datasets (CosmosQA, DREAM, MCTest, MCScript, Quail, SCIQ), adicione mappers específicos no pré-processamento para (article, question, answer, distractors[]) e inclua a nova entrada no YAML.

Como estender

- Novos datasets: crie mapeadores em `preprocess/prepare_dataset.py` seguindo o padrão do RACE.
- Negative sampling: pode-se tornar o mask-filling mais fiel (substituições múltiplas, restrições por semântica), testar outros embeddings/MLM, adicionar ablations (remover Q/Ans/Art) e comparar com métricas como BLEU/BLEURT.

