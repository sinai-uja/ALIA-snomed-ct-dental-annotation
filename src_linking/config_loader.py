"""
Módulo de carga y validación de configuración
"""
import os
import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(path: str, base_dir: str = None) -> Dict[str, Any]:
    """
    Carga configuración desde archivo YAML.
    Opcionalmente resuelve rutas relativas respecto a base_dir.
    
    Args:
        path: Ruta del archivo YAML
        base_dir: Directorio base para resolver rutas relativas
    
    Returns:
        Diccionario con la configuración
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Resolver rutas relativas si se proporciona base_dir
    if base_dir and cfg.get("paths"):
        base_dir = str(Path(base_dir).resolve())
        paths = cfg["paths"]
        for key, value in paths.items():
            if key == "index_name":
                continue
            if isinstance(value, str) and not value.startswith("/"):
                paths[key] = str((Path(base_dir) / value).resolve())
    
    return cfg


def resolve_device(device_str: str) -> str:
    """
    Resuelve el dispositivo especificado.
    
    Args:
        device_str: "auto", "cuda" o "cpu"
    
    Returns:
        Dispositivo resuelto
    """
    if device_str == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def validate_paths(cfg: Dict[str, Any]) -> bool:
    """
    Valida que las rutas en la configuración existan.
    
    Args:
        cfg: Configuración
    
    Returns:
        True si todas las rutas existen
    """
    paths = cfg.get("paths", {})
    critical_paths = ["kb_csv", "input_jsonl", "ner_model", "embed_model"]
    
    missing = []
    for path_key in critical_paths:
        if path_key in paths:
            path = paths[path_key]
            if not os.path.exists(path):
                missing.append(f"{path_key}: {path}")
    
    if missing:
        print("[ERROR] Rutas faltantes:")
        for m in missing:
            print(f"  - {m}")
        return False
    
    return True
