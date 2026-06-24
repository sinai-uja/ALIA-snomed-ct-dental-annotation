"""
Pipeline completo de Entity Linking Híbrido (RRF)

Orquesta: NER -> Entity Linking Híbrido (RRF) sobre KB (CSV)

Búsqueda Híbrida 3-Way RRF:
  - Semántico (FAISS)
  - Léxico (TF-IDF)
  - Morfológico (Levenshtein local)

Fusión: Reciprocal Rank Fusion (RRF) Ponderado

Dependencias:
  pip install transformers torch pandas numpy faiss-gpu-cu12 tqdm pyyaml scikit-learn
"""
import argparse
import os
import sys
from pathlib import Path

from config_loader import load_config, resolve_device, validate_paths
from semantic_index import SentenceEmbedder, SemanticIndex
from hybrid_searcher import HybridSearcher
from entity_linker import load_knowledge_base
from ner_pipeline import build_ner_pipeline
from evaluate_module import process_jsonl, evaluate_linking, run_demo


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Pipeline de Entity Linking Híbrido RRF"
    )
    parser.add_argument(
        "--config",
        default=str(project_root / "config" / "config_linking.yaml"),
        help="Ruta del archivo YAML de configuración",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Directorio base para resolver rutas relativas (por defecto: directorio actual)",
    )
    args = parser.parse_args()

    # Determinar directorio base (por defecto, la raíz del proyecto)
    base_dir = str(Path(args.base_dir).resolve()) if args.base_dir else str(project_root)
    config_path = (
        str((Path(base_dir) / args.config).resolve())
        if not os.path.isabs(args.config)
        else args.config
    )

    print("[Init] Iniciando Pipeline de Entity Linking Híbrido RRF...")
    if not os.path.isfile(config_path):
        print(f"[ERROR] YAML de configuración no encontrado: {config_path}")
        sys.exit(1)

    # Cargar configuración
    cfg = load_config(config_path, base_dir=base_dir)

    # Validar rutas
    if not validate_paths(cfg):
        sys.exit(1)

    # Resolver dispositivo
    device = resolve_device(cfg["misc"]["device"])
    print(f"[Init] Dispositivo asignado: {device.upper()}")

    # Inicializar embedder
    embed_path = cfg["paths"].get("embed_model") or cfg["paths"].get("ner_model")
    embedder = SentenceEmbedder(
        model_path=embed_path,
        max_length=cfg["inference"]["max_length"],
        device=device,
    )

    # Cargar knowledge base
    kb_df = load_knowledge_base(cfg)

    # Construir índice semántico
    sem_index = SemanticIndex(embedder)
    index_name = cfg["paths"].get("index_name", "default_index")
    kb_text_col = cfg.get("kb", {}).get("text_col", "texto")
    sem_index.build(
        kb_df,
        text_col=kb_text_col,
        batch_size=cfg["inference"]["batch_size_embed"],
        cache_dir=cfg["paths"].get("embeddings_cache"),
        index_name=index_name,
    )

    # Construir buscador híbrido
    hybrid_searcher = HybridSearcher(sem_index, cfg)
    hybrid_searcher.build(kb_df, text_col=kb_text_col)

    # Construir pipeline NER
    ner_pipe = build_ner_pipeline(cfg["paths"]["ner_model"], device=device)

    # Ejecutar demo o evaluación
    if cfg["misc"].get("demo", False):
        run_demo(ner_pipe, hybrid_searcher, cfg)
    else:
        stats = process_jsonl(ner_pipe, hybrid_searcher, cfg)
        evaluate_linking(stats)


if __name__ == "__main__":
    main()
