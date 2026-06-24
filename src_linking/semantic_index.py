"""
Módulo de índice semántico basado en FAISS
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


class SentenceEmbedder:
    """
    Genera embeddings de texto usando transformers.
    """

    def __init__(self, model_path: str, max_length: int = 128, device: str = "cpu"):
        """
        Inicializa el embedder.
        
        Args:
            model_path: Ruta del modelo transformers
            max_length: Longitud máxima de secuencias
            device: "cuda" o "cpu"
        """
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.max_length = max_length

        if (
            device == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            self.dtype, dtype_name = torch.bfloat16, "bfloat16"
        else:
            self.dtype, dtype_name = torch.float32, "float32"

        print(
            f"[Embedder] Cargando desde: {model_path} (device={device}, dtype={dtype_name})"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=self.dtype, local_files_only=True
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Codifica una lista de textos a embeddings.
        
        Args:
            texts: Lista de textos
            batch_size: Tamaño de batch para procesamiento
            show_progress: Mostrar barra de progreso
        
        Returns:
            Array de embeddings (N, D)
        """
        all_emb = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", unit="batch")
        use_autocast = self.device == "cuda" and self.dtype == torch.bfloat16

        for start in iterator:
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.autocast(
                device_type=self.device, dtype=self.dtype, enabled=use_autocast
            ):
                out = self.model(**enc)
                hidden = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
            all_emb.append(pooled.float().cpu().numpy())
        return np.vstack(all_emb)


class SemanticIndex:
    """
    Índice semántico FAISS con caché persistente.
    """

    def __init__(self, embedder: SentenceEmbedder):
        """Inicializa el índice semántico."""
        self.embedder = embedder
        self.index = None
        self.kb_df: Optional[pd.DataFrame] = None
        self.kb_emb: Optional[np.ndarray] = None
        self.use_gpu = embedder.device == "cuda"

    @staticmethod
    def _cache_paths(cache_dir: str, index_name: str):
        """Retorna rutas de caché para embeddings e índice."""
        # Evita que un index_name absoluto anule cache_dir.
        safe_index_name = Path(index_name).name or "faiss_index"
        base = Path(cache_dir) / safe_index_name
        return base.with_suffix(".npy"), base.with_suffix(".faiss")

    def _setup_faiss_gpu(self, cpu_index):
        """Trasplanta índice CPU a GPU si está disponible."""
        import faiss

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                return faiss.index_cpu_to_gpu(res, 0, cpu_index)
            except Exception:
                return cpu_index
        return cpu_index

    def build(
        self,
        kb_df: pd.DataFrame,
        text_col: str,
        batch_size: int = 64,
        cache_dir: Optional[str] = None,
        index_name: str = "faiss_index",
    ):
        """
        Construye el índice semántico.
        
        Args:
            kb_df: DataFrame de knowledge base
            text_col: Columna de texto
            batch_size: Tamaño de batch
            cache_dir: Directorio de caché
            index_name: Nombre del índice
        """
        import faiss

        self.kb_df = kb_df.reset_index(drop=True)
        texts = (
            kb_df["search_text"].tolist()
            if "search_text" in kb_df.columns
            else kb_df[text_col].tolist()
        )

        if cache_dir:
            npy_path, faiss_path = self._cache_paths(cache_dir, index_name)
            if npy_path.exists() and faiss_path.exists():
                print(f"[Cache] Leyendo índice semántico: {faiss_path.name}...")
                emb = np.load(str(npy_path))
                cpu_index = faiss.read_index(str(faiss_path))
                self.kb_emb = emb
                self.index = self._setup_faiss_gpu(cpu_index)
                return

        print(
            f"[Index] Generando embeddings semánticos para {len(texts)} entradas..."
        )
        emb = self.embedder.encode(texts, batch_size=batch_size, show_progress=True)
        emb = self._l2_normalize(emb).astype(np.float32)

        self.kb_emb = emb
        dim = emb.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)
        self.index = self._setup_faiss_gpu(cpu_index)
        self.index.add(emb)

        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            npy_path, faiss_path = self._cache_paths(cache_dir, index_name)
            np.save(str(npy_path), emb)
            faiss.write_index(cpu_index, str(faiss_path))

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        """Normaliza vectores a norma L2."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        return x / norms

    def search_with_context(
        self,
        entity_queries: List[str],
        context_queries: List[str],
        alpha: float,
        beta: float,
        pool_size: int,
        threshold: float,
        batch_size: int = 64,
    ):
        """
        Busca semántica ponderando entidad y contexto.
        
        Args:
            entity_queries: Textos de entidades
            context_queries: Textos de contexto
            alpha: Peso para embeddings de entidad
            beta: Peso para embeddings de contexto
            pool_size: Número de candidatos a recuperar
            threshold: Umbral mínimo de similitud
            batch_size: Tamaño de batch
        
        Returns:
            Lista de listas de candidatos
        """
        emb_entity = self._l2_normalize(
            self.embedder.encode(entity_queries, batch_size=batch_size, show_progress=False)
        ).astype(np.float32)
        emb_context = self._l2_normalize(
            self.embedder.encode(context_queries, batch_size=batch_size, show_progress=False)
        ).astype(np.float32)
        emb_combined = self._l2_normalize(
            alpha * emb_entity + beta * emb_context
        ).astype(np.float32)

        scores_ret, indices_ret = self.index.search(emb_entity, pool_size)
        results = []
        for i, (row_scores, row_idx) in enumerate(zip(scores_ret, indices_ret)):
            cands = []
            for score_ent, idx in zip(row_scores, row_idx):
                if idx < 0 or float(score_ent) < threshold:
                    continue
                score_ctx = float(np.dot(emb_combined[i], self.kb_emb[idx]))
                cands.append({"idx": int(idx), "score_sem": round(score_ctx, 4)})
            results.append(cands)
        return results

    def search(
        self,
        queries: List[str],
        pool_size: int,
        threshold: float,
        batch_size: int = 64,
    ):
        """
        Búsqueda semántica directa.
        
        Args:
            queries: Textos de consulta
            pool_size: Número de candidatos
            threshold: Umbral mínimo
            batch_size: Tamaño de batch
        
        Returns:
            Lista de listas de candidatos
        """
        emb = self._l2_normalize(
            self.embedder.encode(queries, batch_size=batch_size, show_progress=False)
        ).astype(np.float32)
        scores, indices = self.index.search(emb, pool_size)
        results = []
        for row_scores, row_idx in zip(scores, indices):
            cands = []
            for score, idx in zip(row_scores, row_idx):
                if idx < 0 or float(score) < threshold:
                    continue
                cands.append({"idx": int(idx), "score_sem": round(float(score), 4)})
            results.append(cands)
        return results
