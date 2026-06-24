# DOCUMENTATION

## 1. Project Objective
This deliverable implements a two-level clinical processing pipeline:

- NER Evaluation (entity extraction) over a dataset in JSONL format.
- Hybrid Entity Linking to map detected entities to SNOMED codes.

The linking process combines three signals:

- Semantic: embeddings + FAISS index.
- Lexical: TF-IDF similarity.
- Morphological: string similarity.

The final candidate fusion is performed using RRF (Reciprocal Rank Fusion).

## 2. Deliverable Structure
Base path: `ENTREGABLE`

- `config/`
- `data/`
- `embedding_cache_desc/`
- `models/`
- `src_e2e/`
- `src_linking/`
- `requirements.txt`
- `README.md`

### 2.1 Configuration
- `config/config_e2e.yaml`: configuration for NER evaluation (`src_e2e/main.py`).
- `config/config_linking.yaml`: configuration for the hybrid NER + Linking pipeline (`src_linking/main.py`).

### 2.2 Data
- `data/test.jsonl`: test set.
- `data/train.jsonl`: training set.
- `data/all.jsonl`: aggregated set.
- `data/snomed_enriched.csv`: enriched SNOMED knowledge base.

### 2.3 Models
- `models/MrBERT-es-ner_model_multiclass_ampere-8192_length/`: multiclass NER model.
- `models/MrBERT-es-ner_model_entity_ampere-8192_length/`: entity model for linking.

### 2.4 Main Code
- `src_e2e/main.py`: NER evaluation and global metrics.
- `src_linking/main.py`: complete orchestrator of the hybrid pipeline.
- `src_linking/config_loader.py`: configuration loading and validation.
- `src_linking/semantic_index.py`: FAISS index construction/reading.
- `src_linking/hybrid_searcher.py`: hybrid retrieval and signal combination.
- `src_linking/entity_linker.py`: linking of detected entities.
- `src_linking/evaluate_module.py`: final evaluation and reporting by method.

## 3. Requirements and Environment
- Python 3.13.
- Dependencies in `requirements.txt`.

Recommended commands:

```bash
pip install -r requirements.txt
```

## 4. How to Run

### 4.1 NER Evaluation
From any folder:

```bash
python main.py
```

With explicit configuration:

```bash
python src_e2e/main.py \
  --config config/config_e2e.yaml
```

### 4.2 Hybrid Linking Pipeline

```bash
python src_linking/main.py
```

With explicit configuration:

```bash
python src_linking/main.py \
  --config config/config_linking.yaml
```

## 5. Functional Flow

### 5.1 NER Flow (`src_e2e/main.py`)
- Loads YAML configuration.
- Resolves paths in the `paths` block.
- Loads tokenizer and NER model.
- Performs inference on JSONL texts.
- Reconstructs BIO spans.
- Computes TP, FP, FN, and global metrics (precision, recall, F1).

### 5.2 Linking Flow (`src_linking/main.py`)
- Loads/validates configuration with `config_loader.py`.
- Selects execution device (`cpu`/`cuda`).
- Loads KB from CSV.
- Builds or reuses embeddings + FAISS index.
- Performs NER on the input text.
- Generates candidates via semantic, lexical, and morphological channels.
- Fuses rankings using RRF.
- Evaluates and reports performance by method (`exact`, `synonym`, `hybrid`, etc.).

### 5.3 Detailed Breakdown of `src_linking` (Module by Module)

`src_linking` implements a hybrid linking pipeline in which entities are first detected in the text, and then the most probable SNOMED code is assigned.

Main components:

- `main.py`: entry point. Orchestrates the entire pipeline execution.
- `config_loader.py`: loads YAML, resolves relative paths, validates critical paths, and detects the device (`auto/cpu/cuda`).
- `ner_pipeline.py`: builds the NER pipeline with Transformers and generates detected entities per document.
- `entity_linker.py`: entity-level linking logic. Attempts exact matching and, if no direct match is found, delegates to hybrid search.
- `semantic_index.py`: generates embeddings, builds the FAISS index, and manages the persistent cache (`.npy` + `.faiss`).
- `lexical_search.py`: lexical search (TF-IDF/text normalization) to retrieve candidates by surface similarity.
- `hybrid_searcher.py`: combines semantic, lexical, and morphological candidates and reranks them using RRF.
- `text_processing.py`: text normalization and label parsing utilities.
- `evaluate_module.py`: computation of global metrics and breakdown by linking method (`exact`, `synonym`, `hybrid`, `no_link`, `other`).
- `evaluate.py`: alternative script for selective evaluation of the NER/linking model for comparative tests.

Internal execution order (summary):

1. Load configuration (`config_linking.yaml`) and resolve paths.
2. Load KB (`snomed_enriched.csv`) and prepare search columns.
3. Build or load embeddings cache and FAISS index.
4. Run NER on `input_jsonl` to obtain detected spans.
5. For each entity, retrieve candidates via semantic/lexical/morphological channels.
6. Fuse candidates with RRF and select the best code.
7. Aggregate results and compute final linking metrics.

Expected output of `src_linking`:

- Code predictions per detected entity.
- End2End quality metrics (NER + linking).
- Reusable caches to accelerate future runs.

## 6. Key Configuration

### 6.1 `config/config_e2e.yaml`
Relevant fields:
- `paths.ner_model`
- `paths.input_jsonl`
- `inference.batch_size_ner`
- `inference.max_length`

Important note:
- In this project, relative paths in this YAML are interpreted relative to the directory of the YAML itself (`config/`).
- Correct example: `../models/...` and `../data/...`.

### 6.2 `config/config_linking.yaml`
Relevant fields:
- `paths.ner_model`
- `paths.embed_model`
- `paths.kb_csv`
- `paths.input_jsonl`
- `paths.output_jsonl`
- `paths.embeddings_cache`
- `paths.index_name`
- `kb.text_col`, `kb.code_col`, `kb.csv_sep`
- `search.semantic_threshold`, `search.lexical_threshold`, `search.rrf_k`

## 7. Cache and FAISS Index
- Cache folder: `embedding_cache_desc/`.
- Expected files for `index_name: snomed_hybrid_v1`:
  - `embedding_cache_desc/snomed_hybrid_v1.npy`
  - `embedding_cache_desc/snomed_hybrid_v1.faiss`

Current behavior:
- `index_name` is treated as a logical identifier for the index, not an absolute path.
- The index is saved within `embeddings_cache`.

## 8. System Outputs
- Metrics and traces printed to the console during NER and linking runs.
- FAISS/embedding cache files in `embedding_cache_desc/`.
- Output files defined by the linking configuration (`paths.output_jsonl`).

## 9. Common Errors and Quick Solutions

### 9.1 `FileNotFoundError` in `test.jsonl`
Cause:
- Relative path resolved incorrectly due to running from another folder or an incorrect value in the YAML.

Solution:
- Verify that `config/config_e2e.yaml` uses `../data/test.jsonl`.

### 9.2 `FileNotFoundError` in `label_map.json`
Cause:
- `paths.ner_model` pointing to `./models/...` from `config/`.

Solution:
- Use `../models/...` in `config/config_e2e.yaml`.

### 9.3 `KeyError: 'kb'` or `KeyError: 'embed_model'`
Cause:
- Incomplete configuration for the linking pipeline.

Solution:
- Run linking with `config/config_linking.yaml`.
- Confirm that `paths`, `kb`, `search`, `inference`, and `misc` blocks exist.

### 9.4 FAISS Saved Outside the Expected Folder
Cause:
- `index_name` treated as a path.

Solution:
- Keep `index_name` as a simple name (e.g., `snomed_hybrid_v1`).
- Verify `paths.embeddings_cache`.

## 10. Delivery and Reproducibility
To reproduce minimal results in a clean environment:

```bash
pip install -r requirements.txt
python src_e2e/main.py
python src_linking/main.py
```

This documents what each module does, how to run it, and how to resolve the most frequent operational issues.