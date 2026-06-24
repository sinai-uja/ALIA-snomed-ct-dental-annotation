"""
Módulo de búsqueda híbrida con Reciprocal Rank Fusion (RRF)
"""
import numpy as np
import pandas as pd
import difflib
from typing import Dict, List, Any, Optional
from semantic_index import SemanticIndex
from lexical_search import LexicalSearcher


class HybridSearcher:
    """
    Buscador híbrido 3-Way RRF: Semántico + Léxico + Morfológico.
    """

    def __init__(self, semantic_index: SemanticIndex, cfg: Dict):
        """
        Inicializa el buscador híbrido.
        
        Args:
            semantic_index: Índice semántico FAISS
            cfg: Diccionario de configuración
        """
        self.sem_index = semantic_index
        self.cfg = cfg
        self.rrf_k = cfg["search"].get("rrf_k", 60)

        self.lexical_searcher = LexicalSearcher(
            lex_threshold=cfg["search"].get("lexical_threshold", 0.1)
        )

    def build(self, kb_df: pd.DataFrame, text_col: str):
        """
        Construye los índices léxicos.
        
        Args:
            kb_df: DataFrame de knowledge base
            text_col: Columna de texto
        """
        self.lexical_searcher.build(kb_df, text_col)

    def search_rrf(
        self,
        entity_queries: List[str],
        context_queries: Optional[List[str]],
        alpha: float,
        beta: float,
        top_k: int,
        threshold_sem: float,
        text_col: str,
        code_col: str,
        batch_size: int = 64,
    ) -> List[List[Dict[str, Any]]]:
        """
        Búsqueda RRF ponderada con 3 canales.
        
        Args:
            entity_queries: Textos de entidades
            context_queries: Textos de contexto (opcional)
            alpha: Peso de entidad
            beta: Peso de contexto
            top_k: Número de candidatos finales
            threshold_sem: Umbral semántico
            text_col: Columna de texto
            code_col: Columna de código
            batch_size: Tamaño de batch
        
        Returns:
            Lista de listas de candidatos ordenados por RRF
        """
        pool_factor = self.cfg["search"].get("hybrid_pool_factor", 10)
        pool_size = max(20, top_k * pool_factor)
        kb_cfg = self.cfg.get("kb", {})
        def_col = kb_cfg.get("def_col")
        syn_col = kb_cfg.get("syn_col")

        # 1. Canal Semántico
        if context_queries:
            sem_results = self.sem_index.search_with_context(
                entity_queries,
                context_queries,
                alpha,
                beta,
                pool_size,
                threshold_sem,
                batch_size,
            )
        else:
            sem_results = self.sem_index.search(
                entity_queries, pool_size, threshold_sem, batch_size
            )

        # 2. Canal Léxico
        lex_sim_matrix = self.lexical_searcher.search(entity_queries, pool_size)

        results = []
        for i, sem_cands in enumerate(sem_results):
            rrf_scores = {}
            debug_info = {}
            query_str = entity_queries[i]

            # -- A) Semántico --
            sem_cands = sorted(
                sem_cands, key=lambda x: x["score_sem"], reverse=True
            )
            for rank, cand in enumerate(sem_cands):
                idx = cand["idx"]
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (
                    self.rrf_k + rank + 1
                )
                debug_info[idx] = {
                    "sem_rank": rank + 1,
                    "lex_rank": "-",
                    "morph_rank": "-",
                }

            # -- B) Léxico --
            row_sim = lex_sim_matrix[i]
            top_lex_idx = np.argsort(row_sim)[::-1][:pool_size]
            lex_rank = 1
            for idx in top_lex_idx:
                sim = row_sim[idx]
                if sim < self.lexical_searcher.lex_threshold:
                    continue
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (
                    self.rrf_k + lex_rank
                )
                if idx not in debug_info:
                    debug_info[idx] = {
                        "sem_rank": "-",
                        "lex_rank": lex_rank,
                        "morph_rank": "-",
                    }
                else:
                    debug_info[idx]["lex_rank"] = lex_rank
                lex_rank += 1

            # -- C) Morfológico (Levenshtein local) --
            candidate_indices = list(debug_info.keys())
            morph_scores = []
            for idx in candidate_indices:
                cand_text = self.sem_index.kb_df.iloc[idx][text_col]
                sim = difflib.SequenceMatcher(None, query_str, cand_text).ratio()

                if syn_col and syn_col in self.sem_index.kb_df.columns:
                    syns_raw = self.sem_index.kb_df.iloc[idx].get("_norm_syns", [])
                    if isinstance(syns_raw, list):
                        for s in syns_raw:
                            sim = max(
                                sim,
                                difflib.SequenceMatcher(None, query_str, s).ratio(),
                            )
                morph_scores.append((idx, sim))

            morph_scores = sorted(morph_scores, key=lambda x: x[1], reverse=True)
            morph_rank = 1
            for idx, sim in morph_scores:
                if sim > 0.5:
                    weight = 2.0 if sim >= 0.8 else 1.0
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + weight / (
                        self.rrf_k + morph_rank
                    )
                    debug_info[idx]["morph_rank"] = morph_rank
                    morph_rank += 1

            # 4. Ordenar candidatos finales
            sorted_cands = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
                :top_k
            ]

            final_cands = []
            for idx, rrf_val in sorted_cands:
                row = self.sem_index.kb_df.iloc[idx]
                dbg = debug_info[idx]

                cand = {
                    "codigo": row[code_col],
                    "texto_kb": row[text_col],
                    "score": round(rrf_val, 5),
                    "method": f"RRF(Sem:{dbg['sem_rank']} Lex:{dbg['lex_rank']} Morph:{dbg['morph_rank']})",
                }

                if def_col and def_col in row:
                    cand["definicion"] = row[def_col]
                if syn_col and syn_col in row:
                    cand["sinonimos"] = row[syn_col]
                final_cands.append(cand)

            results.append(final_cands)

        return results
