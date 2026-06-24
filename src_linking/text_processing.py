"""
Módulo de procesamiento y normalización de texto
"""
import unicodedata
import difflib
from typing import List, Dict, Any, Optional


NORM_VERSION = "v3"


def normalize_text(text: str) -> str:
    """
    Normaliza texto: minúsculas + strip + elimina diacríticos.
    
    Args:
        text: Texto a normalizar
    
    Returns:
        Texto normalizado
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def extract_window(text: str, start: int, end: int, window: int = 60) -> str:
    """
    Extrae una ventana de `window` caracteres a cada lado de la entidad.
    
    Args:
        text: Texto completo
        start: Posición inicial de la entidad
        end: Posición final de la entidad
        window: Caracteres a cada lado
    
    Returns:
        Ventana normalizada
    """
    left = text[max(0, start - window) : start].strip()
    entity = text[start:end].strip()
    right = text[end : end + window].strip()

    parts = []
    if left:
        parts.append(left)
    parts.append(entity)
    if right:
        parts.append(right)

    return normalize_text(" ".join(parts))


def singular_candidates(word: str) -> List[str]:
    """
    Genera variaciones morfológicas simples (plurales, género, sufijos).
    
    Args:
        word: Palabra a expandir
    
    Returns:
        Lista de variantes morfológicas
    """
    cands = []
    # Plurales clásicos
    if word.endswith("es") and len(word) > 4:
        cands.append(word[:-2])
    if word.endswith("s") and len(word) > 3:
        cands.append(word[:-1])

    # Género y sufijos comunes
    if word.endswith("a") and len(word) > 3:
        cands.append(word[:-1] + "o")
    if word.endswith("o") and len(word) > 3:
        cands.append(word[:-1] + "a")
    if word.endswith("as") and len(word) > 4:
        cands.append(word[:-2] + "os")
    if word.endswith("os") and len(word) > 4:
        cands.append(word[:-2] + "as")
    if word.endswith("al") and len(word) > 4:
        cands.append(word[:-1])
    if word.endswith("ales") and len(word) > 5:
        cands.append(word[:-2])

    return cands


def parse_label_span(span: Any) -> Optional[Dict[str, Any]]:
    """
    Parsea un span de anotación del dataset.
    
    Args:
        span: Tupla (start, end, label) o similar
    
    Returns:
        Diccionario con start, end, description, code
    """
    if not (isinstance(span, (list, tuple)) and len(span) == 3):
        return None
    start, end, raw_label = span
    try:
        start, end = int(start), int(end)
    except (TypeError, ValueError):
        return None

    raw_label = str(raw_label)
    if " - " in raw_label:
        parts = raw_label.rsplit(" - ", 1)
        description = parts[0].strip()
        code = parts[1].strip()
    else:
        description = raw_label.strip()
        code = None
    return {"start": start, "end": end, "description": description, "code": code}


def parse_labels(label_obj: Any) -> List[Dict[str, Any]]:
    """
    Parsea una lista de anotaciones.
    
    Args:
        label_obj: Lista de spans o None
    
    Returns:
        Lista de diccionarios de spans parseados
    """
    if label_obj is None:
        return []
    if isinstance(label_obj, (list, tuple)):
        return [p for span in label_obj if (p := parse_label_span(span)) is not None]
    return []
