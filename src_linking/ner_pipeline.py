"""
Módulo para pipeline de Named Entity Recognition (NER)
"""
from typing import List, Dict, Any
from tqdm import tqdm


def build_ner_pipeline(model_path: str, device: str):
    """
    Construye un pipeline de NER usando transformers.
    
    Args:
        model_path: Ruta del modelo NER
        device: "cuda" o "cpu"
    
    Returns:
        Pipeline de NER
    """
    from transformers import pipeline

    print(f"[NER] Cargando modelo de extracción NER desde: {model_path}")
    return pipeline(
        "ner",
        model=model_path,
        tokenizer=model_path,
        aggregation_strategy="simple",
        device=0 if device == "cuda" else -1,
    )


def run_ner(
    ner_pipe, texts: List[str], batch_size: int = 32
) -> List[List[Dict]]:
    """
    Ejecuta el pipeline NER en lotes de textos.
    
    Args:
        ner_pipe: Pipeline NER
        texts: Lista de textos
        batch_size: Tamaño de lote
    
    Returns:
        Lista de resultados NER
    """
    all_out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="NER inference"):
        batch = texts[i : i + batch_size]
        results = ner_pipe(batch)
        if results and isinstance(results[0], dict):
            results = [results]
        all_out.extend(results)
    return all_out
