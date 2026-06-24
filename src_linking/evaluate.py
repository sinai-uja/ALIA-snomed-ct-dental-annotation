import json
import torch
import os
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict
from tqdm import tqdm


def load_config(config_path: str, base_dir: str = None) -> Dict[str, Any]:
    """Carga configuración desde YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Resolver rutas relativas
    if base_dir and cfg.get("paths"):
        base_dir = str(Path(base_dir).resolve())
        paths = cfg["paths"]
        for key, value in paths.items():
            if isinstance(value, str) and not value.startswith("/"):
                paths[key] = str((Path(base_dir) / value).resolve())
    
    return cfg

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def get_entities_from_bio(labels: List[str], offsets: List[Any], text: str):
    entities = []
    current_ent = None
    for i, label in enumerate(labels):
        if i >= len(offsets): break
        start, end = offsets[i]
        if start == end: continue 
        if label.startswith("B-"):
            if current_ent: entities.append(current_ent)
            actual_start = start
            while actual_start < end and text[actual_start].isspace(): actual_start += 1
            current_ent = {"label": label[2:], "start": actual_start, "end": end, "text": text[actual_start:end].strip()}
        elif label.startswith("I-") and current_ent:
            if label[2:] == current_ent["label"]:
                current_ent["end"] = end
                current_ent["text"] = text[current_ent["start"]:end].strip()
        else:
            if current_ent: entities.append(current_ent)
            current_ent = None
    if current_ent: entities.append(current_ent)
    return entities

def run_evaluation(cfg: Dict[str, Any]):
    """Ejecuta evaluación del modelo NER."""
    MODEL_PATH = cfg["paths"]["ner_model"]
    TEST_JSONL_PATH = cfg["paths"]["input_jsonl"]
    MAX_LENGTH = cfg["inference"]["max_length"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(f"{MODEL_PATH}/label_map.json", "r") as f:
        id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    raw_data = load_jsonl(TEST_JSONL_PATH)
    stats_per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    print(MODEL_PATH)
    print(f"\n{'='*100}\n{'DETALLE POR DOCUMENTO':^100}\n{'='*100}")

    for doc_idx, doc in enumerate(raw_data):
        text = doc["text"]
        gold_labels = [{"start": g[0], "end": g[1], "label": g[2]} if isinstance(g, list) else g for g in doc.get("label", [])]
        
        inputs = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_offsets_mapping=True, return_tensors="pt").to(device)
        offsets = inputs.pop("offset_mapping")[0].tolist()
        
        with torch.no_grad():
            predictions = torch.argmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()

        pred_tags = [id2label[p] for p in predictions]
        detected_ents = get_entities_from_bio(pred_tags, offsets, text)

        print(f"\n📄 DOC ID: {doc.get('id', doc_idx)} | Texto: {text[:70]}...")
        matched_gold_indices = set()
        for det in detected_ents:
            match_idx = next((i for i, g in enumerate(gold_labels) if det["start"] == g["start"] and det["end"] == g["end"] and det["label"] == g["label"]), -1)
            if match_idx != -1:
                stats_per_label[det["label"]]["tp"] += 1
                matched_gold_indices.add(match_idx)
                print(f"  ✔️ OK    | [{det['start']:>4}:{det['end']:<4}] | {det['label']:<30} | '{det['text']}'")
            else:
                stats_per_label[det["label"]]["fp"] += 1
                print(f"  ❌ FP    | [{det['start']:>4}:{det['end']:<4}] | {det['label']:<30} | '{det['text']}' (No en dataset)")

        for i, gold in enumerate(gold_labels):
            if i not in matched_gold_indices:
                stats_per_label[gold["label"]]["fn"] += 1
                print(f"  ⚠️ MISS  | [{gold['start']:>4}:{gold['end']:<4}] | {gold['label']:<30} | '{text[gold['start']:gold['end']]}'")

    # --- CÁLCULO DE MÉTRICAS GLOBALES ---
    g_tp = sum(m['tp'] for m in stats_per_label.values())
    g_fp = sum(m['fp'] for m in stats_per_label.values())
    g_fn = sum(m['fn'] for m in stats_per_label.values())

    total_precision = g_tp / (g_tp + g_fp) if (g_tp + g_fp) > 0 else 0
    total_recall = g_tp / (g_tp + g_fn) if (g_tp + g_fn) > 0 else 0
    total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

    print("\n" + "="*100)
    print(MODEL_PATH)
    print(f"{'RESUMEN GLOBAL DEL MODELO':^100}")
    print("="*100)
    print(f"  PRECISIÓN GLOBAL: {total_precision:.4f} (¿Cuánto de lo que predigo es correcto?)")
    print(f"  RECALL GLOBAL:    {total_recall:.4f} (¿Cuántas del total he encontrado?)")
    print(f"  F1-SCORE GLOBAL:  {total_f1:.4f} (Balance general)")
    print("-" * 100)
    print(f"  Total Aciertos (TP): {g_tp}")
    print(f"  Total Errores (FP):   {g_fp}")
    print(f"  Total No Detectadas (FN): {g_fn}")
    print("="*100 + "\n")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(description="Evaluación de modelo NER")
    parser.add_argument(
        "--config",
        default=str(project_root / "config" / "config_ner_eval.yaml"),
        help="Ruta del archivo YAML de configuración (default: config_ner_eval.yaml para NER simple)",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Directorio base para resolver rutas relativas",
    )
    args = parser.parse_args()

    base_dir = str(Path(args.base_dir).resolve()) if args.base_dir else str(project_root)
    config_path = (
        str((Path(base_dir) / args.config).resolve())
        if not os.path.isabs(args.config)
        else args.config
    )

    if not os.path.isfile(config_path):
        print(f"[ERROR] YAML de configuración no encontrado: {config_path}")
        exit(1)

    cfg = load_config(config_path, base_dir=base_dir)
    run_evaluation(cfg)