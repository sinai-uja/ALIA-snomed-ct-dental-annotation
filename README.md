# SNOMED Entity Linking (NER + Hybrid Linking)

Project for medical entity extraction (NER) and linking against SNOMED using hybrid search:
- semantic (embeddings + FAISS)
- lexical (TF-IDF)
- morphological (string similarity)
- fusion by RRF (Reciprocal Rank Fusion)

## Current Status

- Linking pipeline: `src_linking/main.py`
- NER evaluation: `src_e2e/main.py`
- Configurations in `config/`

## Structure

```text
ALIA-snomed-ct-dental-annotation/
├── config/
│   ├── config_linking.yaml      # Linking config
│   └── config_e2e.yaml          # End-to-end (NER evaluation) config
├── data/
│   ├── test.jsonl
│   ├── train.jsonl
│   ├── all.jsonl
│   └── snomed_enriched.csv
├── models/
│   ├── MrBERT-es-ner_model_entity_ampere-8192_length/
│   └── MrBERT-es-ner_model_multiclass_ampere-8192_length/
├── src_linking/
│   ├── main.py
│   ├── config_loader.py
│   ├── entity_linker.py
│   ├── hybrid_searcher.py
│   ├── lexical_search.py
│   ├── semantic_index.py
│   ├── ner_pipeline.py
│   └── evaluate_module.py
├── src_e2e/
│   └── main.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.13+
- Environment with `torch`, `transformers`, `pandas`, `faiss`, `scikit-learn`

Recommended installation:

```bash
pip install -r requirements.txt
```

## Configuration

### 1) Full Pipeline (Linking)

File: `config/config_linking.yaml`

Includes:
- model and data paths
- hybrid search parameters (`semantic_threshold`, `lexical_threshold`, `rrf_k`, etc.)
- embeddings cache

### 2) NER Evaluation

File: `config/config_e2e.yaml`

Includes only:
- NER model
- test dataset
- NER inference parameters

## Execution

### Option A: Full Pipeline (NER + Linking)

From the project root:

```bash
python src_linking/main.py
```

Optional, forcing a specific config:

```bash
python src_linking/main.py --config config/config_linking.yaml --base-dir .
```

### Option B: NER Evaluation Only

```bash
python src_e2e/main.py --config config/config_e2e.yaml
```

If running from `src_e2e/`:

```bash
cd src_e2e
python main.py
```

## Outputs

- Embeddings cache: `embedding_cache_desc/`
- FAISS index: generated in the cache folder or the configured folder
- Linking output: `output/` (based on configuration)

## Current Results

### Linking

- NER Accuracy (Spans): `5818/6022` (`96.61%`)
- End2End Linking Accuracy: `5742/6022` (`95.35%`)
- Conditioned Linking Accuracy (on NER hits): `5742/5818` (`98.69%`)
- NER False Positives: `304`

### Global End-to-End Metrics (NER)

- Global Precision: `0.9668`
- Global Recall: `0.9573`
- Global F1: `0.9620`
- TP: `5765`
- FP: `198`
- FN: `257`

## Technical Notes

- The `UNEXPECTED` warning when loading ModernBERT may appear when checkpoint heads/tasks change.
- The KB parser has been hardened for CSV delimiters (and can be set via `kb.csv_sep` in the config).

## Quick Troubleshooting

### CSV Parser Error

Make sure `kb.csv_sep` in `config/config_linking.yaml` matches the file format (`","` or `";"`).

## Internal Versioning

- `src_linking/`: Linking version
- `src_e2e/`: End-to-end / NER evaluation version
