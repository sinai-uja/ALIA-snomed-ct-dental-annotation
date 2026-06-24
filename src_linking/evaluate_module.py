"""
Módulo de evaluación y estadísticas
"""
import json
from typing import Dict, List, Any
from tqdm import tqdm
from text_processing import parse_labels
from ner_pipeline import run_ner
from entity_linker import link_entities


def _init_bucket_stats() -> Dict[str, Any]:
    """Inicializa estadísticas para un bucket de métodos."""
    return {"correct": 0, "wrong": 0, "wrong_no_code": 0, "fp": 0}


def _push_example(
    store: Dict[str, List[Dict[str, Any]]], key: str, item: Dict[str, Any], max_n: int
):
    """Añade un ejemplo a la colección."""
    lst = store.setdefault(key, [])
    if len(lst) < max_n:
        lst.append(item)


def method_bucket(link_method: str) -> str:
    """Clasifica método de linking en buckets."""
    if not link_method:
        return "no_link"
    if link_method == "exact_match":
        return "exact"
    if link_method == "synonym_exact_match":
        return "synonym"
    if isinstance(link_method, str) and link_method.startswith("RRF"):
        return "hybrid"
    return "other"


def process_jsonl(ner_pipe, hybrid_searcher, cfg: Dict):
    """
    Procesa dataset JSONL y devuelve estadísticas de linking.
    
    Args:
        ner_pipe: Pipeline NER
        hybrid_searcher: Buscador híbrido
        cfg: Configuración
    
    Returns:
        Diccionario con estadísticas
    """
    input_path = cfg["paths"]["input_jsonl"]
    ner_bs = cfg["inference"]["batch_size_ner"]
    max_examples = cfg.get("misc", {}).get("max_match_examples", 8)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(
        f"\n[Procesamiento] Corriendo inferencia NER sobre {len(records)} documentos..."
    )
    texts = [r["text"] for r in records]
    ner_results = run_ner(ner_pipe, texts, batch_size=ner_bs)

    stats = {
        "total_gold": 0,
        "ner_hits": 0,
        "link_hits": 0,
        "by_method": {
            "exact": _init_bucket_stats(),
            "synonym": _init_bucket_stats(),
            "hybrid": _init_bucket_stats(),
            "no_link": _init_bucket_stats(),
            "other": _init_bucket_stats(),
        },
        "examples": {"correct": {}, "wrong": {}, "fp": {}},
    }

    for doc_id, (record, ner_out) in enumerate(zip(records, ner_results)):
        raw_text = record["text"]
        linked = link_entities(ner_out, hybrid_searcher, cfg, raw_text=raw_text)
        gold_spans = parse_labels(record.get("label", []))
        stats["total_gold"] += len(gold_spans)
        matched_gold = set()

        for ent in linked:
            p_span = (ent["start"], ent["end"])
            p_code, p_text, p_meth = (
                ent["linked_code"],
                ent["linked_text"],
                ent["link_method"],
            )
            p_score = ent.get("link_score")
            bucket = method_bucket(p_meth)

            found_gold = None
            for i, g in enumerate(gold_spans):
                if p_span == (g["start"], g["end"]):
                    found_gold = g
                    if i not in matched_gold:
                        stats["ner_hits"] += 1
                        matched_gold.add(i)
                    break

            if found_gold is None:
                stats["by_method"][bucket]["fp"] += 1
                _push_example(
                    stats["examples"]["fp"],
                    bucket,
                    {
                        "doc_id": doc_id,
                        "span": p_span,
                        "word": ent["word"],
                        "pred_code": p_code,
                        "pred_text": p_text,
                        "method": p_meth,
                        "score": p_score,
                    },
                    max_examples,
                )
                continue

            is_correct = p_code is not None and p_code == found_gold["code"]

            if is_correct:
                stats["link_hits"] += 1
                stats["by_method"][bucket]["correct"] += 1
                _push_example(
                    stats["examples"]["correct"],
                    bucket,
                    {
                        "doc_id": doc_id,
                        "span": p_span,
                        "word": ent["word"],
                        "pred_code": p_code,
                        "gold_code": found_gold["code"],
                        "method": p_meth,
                        "score": p_score,
                    },
                    max_examples,
                )
            else:
                stats["by_method"][bucket]["wrong"] += 1
                if p_code is None:
                    stats["by_method"][bucket]["wrong_no_code"] += 1
                _push_example(
                    stats["examples"]["wrong"],
                    bucket,
                    {
                        "doc_id": doc_id,
                        "span": p_span,
                        "word": ent["word"],
                        "pred_code": p_code,
                        "pred_text": p_text,
                        "gold_code": found_gold["code"],
                        "gold_desc": found_gold.get("description"),
                        "method": p_meth,
                        "score": p_score,
                    },
                    max_examples,
                )

    return stats


def evaluate_linking(stats: Dict):
    """
    Imprime reporte detallado de evaluación.
    
    Args:
        stats: Estadísticas de linking
    """
    tg, nh, lh = stats["total_gold"], stats["ner_hits"], stats["link_hits"]
    total_fps = sum(b["fp"] for b in stats["by_method"].values())

    print(
        "\n"
        + "=" * 85
        + f"\n{'REPORT FINAL DE ENTITY LINKING':^85}\n"
        + "=" * 85
    )
    print(f"1. Precisión NER (Spans):     {nh}/{tg} ({(nh/tg*100) if tg else 0:.2f}%)")
    print(
        f"2. Precisión Linking (End2End): {lh}/{tg} ({(lh/tg*100) if tg else 0:.2f}%)"
    )
    print(
        f"3. Precisión Linking Condicionada (Solo NER Hits): {lh}/{nh} ({(lh/nh*100) if nh else 0:.2f}%)"
    )
    print(f"4. Falsos Positivos NER (Extraídos sin gold): {total_fps}")
    print("-" * 85)

    print("\nDESGLOSE DE RENDIMIENTO POR MÉTODO:")
    header = f"{'MÉTODO':<12} | {'ACIERTOS':>8} | {'FALLOS':>8} | {'% DEL MÉTODO':>12} | {'% DEL TOTAL':>12}"
    print(header)
    print("-" * len(header))

    total_correct, total_wrong = 0, 0
    for bucket in ["exact", "synonym", "hybrid", "no_link", "other"]:
        b = stats["by_method"][bucket]
        correct, wrong = b["correct"], b["wrong"]
        denom = correct + wrong
        total_correct += correct
        total_wrong += wrong

        acc_method = (correct / denom * 100.0) if denom else 0.0
        acc_total = (correct / lh * 100.0) if lh else 0.0

        if denom > 0:
            print(
                f"{bucket:<12} | {correct:>8d} | {wrong:>8d} | {acc_method:>11.1f}% | {acc_total:>11.1f}%"
            )

    print("-" * len(header))
    total_acc = (
        (total_correct / (total_correct + total_wrong) * 100)
        if (total_correct + total_wrong)
        else 0
    )
    print(
        f"{'TOTAL':<12} | {total_correct:>8d} | {total_wrong:>8d} | {total_acc:>11.1f}% |      100.0%"
    )
    print("\n" + "=" * 85)

    def _print_examples(kind: str, title: str):
        print(f"\n{title}:")
        any_printed = False
        for bucket in ["exact", "synonym", "hybrid", "no_link", "other"]:
            ex = stats["examples"][kind].get(bucket, [])
            if not ex:
                continue
            any_printed = True
            print(f"\n  [{bucket.upper()}] ({len(ex)} ejemplos guardados):")
            for e in ex:
                score_str = (
                    f"{e.get('score'):.5f}" if e.get("score") is not None else "N/A"
                )

                if kind == "correct":
                    print(
                        f"    ✅ '{e['word']}' -> Code: {e['pred_code']} | Score: {score_str} (Método: {e['method']})"
                    )
                elif kind == "wrong":
                    pred = (
                        f"{e['pred_code']} ({e['pred_text']})"
                        if e["pred_code"]
                        else "None"
                    )
                    print(
                        f"    ⚠️ '{e['word']}' -> PRED: {pred} | Score: {score_str} | REAL: {e['gold_code']} ({e['gold_desc']})  [{e['method']}]"
                    )
                else:
                    print(
                        f"    ❌ '{e['word']}' -> PRED: {e['pred_code']} ({e['pred_text']}) | Score: {score_str} (Falso Positivo NER) [{e['method']}]"
                    )

        if not any_printed:
            print("  (Ningún ejemplo registrado)")

    _print_examples("correct", "EJEMPLOS DE ACIERTOS (Por Método)")
    _print_examples("wrong", "EJEMPLOS DE FALLOS (Por Método)")
    _print_examples(
        "fp", "EJEMPLOS DE FALSOS POSITIVOS (NER extrajo spans sin equivalente en gold)"
    )
    print("\n" + "=" * 85)


def run_demo(ner_pipe, hybrid_searcher, cfg: Dict):
    """
    Ejecuta demo en texto de ejemplo.
    
    Args:
        ner_pipe: Pipeline NER
        hybrid_searcher: Buscador híbrido
        cfg: Configuración
    """
    demo_text = cfg["misc"]["demo_text"]
    print(
        "\n"
        + "=" * 60
        + "\nDEMO\n"
        + "=" * 60
        + f"\nTexto: {demo_text}\n"
    )
    ner_out = ner_pipe(demo_text)
    if isinstance(ner_out, dict):
        ner_out = [ner_out]

    linked = link_entities(ner_out, hybrid_searcher, cfg, raw_text=demo_text)
    for ent in linked:
        print(
            f" ▶ '{ent['word']}'  [{ent['entity_group']}]"
        )
        if ent["linked_code"]:
            print(f"   → cod: {ent['linked_code']} | kb: {ent['linked_text']}")
            if ent.get("linked_syns"):
                print(f"   → sinónimos: {ent['linked_syns']}")
            if ent.get("linked_def"):
                print(f"   → def: {ent['linked_def'][:80]}...")
            print(f"   → via: {ent['link_method']} | score: {ent['link_score']:.5f}")
        else:
            print("   → sin match (threshold no alcanzado)")
        for c in ent["candidates"][1:]:
            print(
                f"      [{c['score']:.5f}] {c['codigo']} | {c['texto_kb']} | {c.get('method','')}"
            )
        print()
