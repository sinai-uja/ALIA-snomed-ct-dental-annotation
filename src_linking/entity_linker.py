"""
Módulo de linking de entidades al knowledge base
"""
import pandas as pd
import csv
from typing import List, Dict, Any, Optional
from text_processing import normalize_text, extract_window
from lexical_search import lexical_match
from hybrid_searcher import HybridSearcher


def link_entities(
    ner_entities: List[Dict],
    hybrid_searcher: HybridSearcher,
    cfg: Dict,
    raw_text: str = "",
) -> List[Dict]:
    """
    Linkea entidades NER al knowledge base usando búsqueda híbrida.
    
    Args:
        ner_entities: Entidades extraídas por NER
        hybrid_searcher: Buscador híbrido
        cfg: Configuración
        raw_text: Texto original (para contexto)
    
    Returns:
        Lista de entidades linkeadas
    """
    if not ner_entities:
        return []

    top_k = cfg["search"]["top_k"]
    threshold_sem = cfg["search"].get(
        "semantic_threshold", cfg["search"].get("threshold", 0.75)
    )
    text_col = cfg["kb"]["text_col"]
    code_col = cfg["kb"]["code_col"]
    def_col = cfg["kb"].get("def_col")
    syn_col = cfg["kb"].get("syn_col")
    embed_bs = cfg["inference"]["batch_size_embed"]
    alpha = cfg["search"].get("context_alpha", 0.9)
    beta = cfg["search"].get("context_beta", 0.1)
    context_window = cfg["search"].get("context_window", 80)
    use_context = beta > 0.0 and raw_text

    queries_entity = [normalize_text(e["word"].strip()) for e in ner_entities]

    # Intenta match léxico primero (rápido)
    lexical_hits = [
        lexical_match(
            q,
            hybrid_searcher.sem_index.kb_df,
            text_col,
            code_col,
            def_col,
            syn_col,
        )
        for q in queries_entity
    ]
    sem_indices = [i for i, h in enumerate(lexical_hits) if h is None]
    sem_results: Dict[int, List[Dict]] = {}

    # Para las que no tienen match léxico, usa búsqueda semántica
    if sem_indices:
        entity_queries = [queries_entity[i] for i in sem_indices]
        context_queries = None

        if use_context:
            context_queries = []
            for i in sem_indices:
                e = ner_entities[i]
                s, en = e.get("start"), e.get("end")
                ctx = (
                    extract_window(raw_text, s, en, window=context_window)
                    if (s is not None and en is not None)
                    else queries_entity[i]
                )
                context_queries.append(ctx)

        raw = hybrid_searcher.search_rrf(
            entity_queries=entity_queries,
            context_queries=context_queries,
            alpha=alpha,
            beta=beta,
            top_k=top_k,
            threshold_sem=threshold_sem,
            text_col=text_col,
            code_col=code_col,
            batch_size=embed_bs,
        )

        for idx, cands in zip(sem_indices, raw):
            sem_results[idx] = cands

    all_candidates = []
    for i in range(len(ner_entities)):
        if lexical_hits[i] is not None:
            all_candidates.append([lexical_hits[i]])
        else:
            all_candidates.append(sem_results.get(i, []))

    linked = []
    for entity, cands in zip(ner_entities, all_candidates):
        raw_start, raw_end = entity.get("start"), entity.get("end")
        word = entity["word"].strip()

        real_start = raw_start
        if raw_start is not None and raw_end is not None and raw_text:
            while real_start < raw_end and raw_text[real_start] == " ":
                real_start += 1

        linked.append(
            {
                "word": word,
                "entity_group": entity["entity_group"],
                "start": real_start,
                "end": raw_end,
                "linked_code": cands[0]["codigo"] if cands else None,
                "linked_text": cands[0]["texto_kb"] if cands else None,
                "linked_def": cands[0].get("description") if cands else None,
                "linked_syns": cands[0].get("synonym") if cands else None,
                "link_score": cands[0]["score"] if cands else None,
                "link_method": cands[0].get("method", "unk") if cands else None,
                "candidates": cands,
            }
        )
    return linked


def load_knowledge_base(cfg: Dict) -> pd.DataFrame:
    """
    Carga y prepara el knowledge base desde CSV.
    
    Args:
        cfg: Configuración
    
    Returns:
        DataFrame de knowledge base con campos normalizados
    """
    csv_path = cfg["paths"]["kb_csv"]
    text_col = cfg["kb"]["text_col"]
    code_col = cfg["kb"]["code_col"]
    def_col = cfg["kb"].get("descripcion_llm")
    syn_col = cfg["kb"].get("sinonimos_llm")
    csv_sep = cfg["kb"].get("csv_sep")

    if not csv_sep:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(8192)
        try:
            csv_sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except Exception:
            csv_sep = ","

    print(f"[KB] Cargando y normalizando KB desde {csv_path} (sep='{csv_sep}') ...")
    df = pd.read_csv(csv_path, sep=csv_sep, dtype=str)
    df = df.dropna(subset=[text_col, code_col])

    df[text_col] = df[text_col].apply(normalize_text)
    df[code_col] = df[code_col].str.strip()

    search_parts = [df[text_col]]

    if def_col and def_col in df.columns:
        print(f"[KB] Incorporando columna de definición: '{def_col}'")
        df[def_col] = df[def_col].fillna("").astype(str).str.strip()
        search_parts.append(df[def_col].apply(normalize_text))

    if syn_col and syn_col in df.columns:
        print(f"[KB] Incorporando columna de sinónimos: '{syn_col}'")
        df[syn_col] = df[syn_col].fillna("").astype(str).str.strip()
        df["_norm_syns"] = df[syn_col].apply(
            lambda x: [normalize_text(s) for s in x.split("|") if s.strip()]
        )
        clean_syns_text = df[syn_col].str.replace(r"\s*\|\s*", " ", regex=True)
        search_parts.append(clean_syns_text.apply(normalize_text))

    df["search_text"] = search_parts[0].str.cat(search_parts[1:], sep=" ", na_rep="")
    print(f"[KB] {len(df)} registros cargados con éxito.")
    return df.reset_index(drop=True)
