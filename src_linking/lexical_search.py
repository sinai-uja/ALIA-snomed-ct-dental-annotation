"""
Módulo de búsqueda léxica y matching
"""
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from text_processing import normalize_text, singular_candidates


def lexical_match(
    query_norm: str,
    kb_df: pd.DataFrame,
    text_col: str,
    code_col: str,
    def_col: Optional[str] = None,
    syn_col: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Match léxico rápido (exact match, morphological variants, synonyms).
    
    Args:
        query_norm: Consulta normalizada
        kb_df: DataFrame de knowledge base
        text_col: Columna de texto
        code_col: Columna de código
        def_col: Columna de definición (opcional)
        syn_col: Columna de sinónimos (opcional)
    
    Returns:
        Diccionario con match o None
    """

    def _first_match(candidate: str, score: float):
        mask_text = kb_df[text_col] == candidate
        mask_syn = pd.Series([False] * len(kb_df), index=kb_df.index)
        if "_norm_syns" in kb_df.columns:
            mask_syn = kb_df["_norm_syns"].apply(lambda syns: candidate in syns)

        mask = mask_text | mask_syn
        if mask.any():
            match_idx = mask.idxmax()
            row = kb_df.loc[match_idx]

            method = (
                "exact_match"
                if row[text_col] == candidate
                else "synonym_exact_match"
            )
            res = {
                "codigo": row[code_col],
                "texto_kb": row[text_col],
                "score": score,
                "method": method,
            }

            if def_col and def_col in kb_df.columns:
                res["definicion"] = row[def_col]
            if syn_col and syn_col in kb_df.columns:
                res["sinonimos"] = row[syn_col]

            return res
        return None

    hit = _first_match(query_norm, 1.0)
    if hit:
        return hit

    for stem in singular_candidates(query_norm):
        hit = _first_match(stem, 0.99)
        if hit:
            return hit

    words = query_norm.split()
    if len(words) > 1:
        from itertools import product as iproduct

        variants = [[w] + singular_candidates(w) for w in words]
        for combo in iproduct(*variants):
            candidate = " ".join(combo)
            if candidate == query_norm:
                continue
            hit = _first_match(candidate, 0.98)
            if hit:
                return hit

    return None


class LexicalSearcher:
    """
    Buscador léxico basado en TF-IDF character n-grams.
    """

    def __init__(self, lex_threshold: float = 0.1):
        """Inicializa el buscador léxico."""
        self.lex_threshold = lex_threshold
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.tfidf_matrix = None

    def build(self, kb_df: pd.DataFrame, text_col: str):
        """
        Entrena el vectorizador TF-IDF.
        
        Args:
            kb_df: DataFrame de knowledge base
            text_col: Columna de texto
        """
        print("[Lexical] Construyendo índice léxico (TF-IDF char_wb)...")
        texts = (
            kb_df["search_text"].tolist()
            if "search_text" in kb_df.columns
            else kb_df[text_col].tolist()
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, entity_queries: List[str], pool_size: int):
        """
        Busca léxica.
        
        Args:
            entity_queries: Lista de consultas
            pool_size: Número máximo de candidatos
        
        Returns:
            Matriz de similitud coseno
        """
        from sklearn.metrics.pairwise import cosine_similarity

        query_vecs = self.vectorizer.transform(entity_queries)
        return cosine_similarity(query_vecs, self.tfidf_matrix)
