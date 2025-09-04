#!/usr/bin/env python3
# hyde_ragfusion.py - improved HyDE + RAG-Fusion
# Features:
#  - PDF ingest & chunking (simple, fast)
#  - Small/faster embedder (MiniLM)
#  - Optional FAISS index (if faiss installed)
#  - BM25 lexical retriever (rank_bm25)
#  - HyDE (local seq2seq generator used to synthesize hypothetical doc)
#  - Query variants + RRF fusion
#  - Answer generation (local FLAN-T5) with inline provenance
#  - Index + metadata persistence

from __future__ import annotations
import os, sys, json, re, math, heapq, argparse, time
from typing import List, Tuple, Dict, Optional

# optional heavy deps
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------------
# Utilities
# -----------------------------
def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def chunk_text(text: str, chunk_size_words: int = 300, overlap_words: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size_words])
        chunks.append(normalize_whitespace(chunk))
        i += chunk_size_words - overlap_words
    return chunks

def save_json(pat: str, obj):
    with open(pat, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(pat: str):
    with open(pat, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# Ingest PDFs -> corpus + meta
# -----------------------------
def load_pdfs(folder: str) -> List[Tuple[str,str]]:
    pairs = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, fn)
        try:
            with fitz.open(path) as doc:
                pages = []
                for p in doc:
                    txt = p.get_text("text") or ""
                    pages.append(txt)
                text = "\n".join(pages)
        except Exception as e:
            print(f"[WARN] Could not read {fn}: {e}")
            continue
        text = normalize_whitespace(text)
        if text:
            pairs.append((fn, text))
    return pairs

def build_corpus_from_folder(folder: str, chunk_size_words=300, overlap_words=50,
                             deduplicate: bool = True) -> Tuple[List[str], List[Dict]]:
    raw = load_pdfs(folder)
    corpus = []
    metas = []
    seen = set()
    for fname, text in raw:
        chunks = chunk_text(text, chunk_size_words, overlap_words)
        for i, ch in enumerate(chunks):
            key = (fname, i, ch[:200])
            if deduplicate and key in seen:
                continue
            seen.add(key)
            corpus.append(ch)
            metas.append({"file": fname, "chunk_id": i})
    return corpus, metas

# -----------------------------
# Model loading (device-aware)
# -----------------------------
def load_models(device: Optional[str] = None,
                embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                gen_model: str = "google/flan-t5-base"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device={device}")
    embedder = SentenceTransformer(embed_model, device=device)
    tok = AutoTokenizer.from_pretrained(gen_model)
    gen = AutoModelForSeq2SeqLM.from_pretrained(gen_model).to(device)
    return embedder, tok, gen, device

# -----------------------------
# Indexing: embeddings + (optional) FAISS
# -----------------------------
def build_embeddings(embedder, corpus: List[str], batch_size=64, normalize=True) -> np.ndarray:
    emb = embedder.encode(corpus, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    if normalize:
        # L2 normalize for cosine via dot-product
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        emb = emb / norms
    return emb.astype("float32")

def build_index(embeddings: np.ndarray):
    if _HAS_FAISS:
        dim = embeddings.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(embeddings)
        return ("faiss", idx)
    else:
        # fallback: brute force numpy search
        return ("numpy", embeddings)

def save_index(kind, index_obj, path_prefix="index/index"):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    if kind == "faiss":
        faiss.write_index(index_obj, path_prefix + ".faiss")
    else:
        np.save(path_prefix + ".npy", index_obj)

def load_index(kind, path_prefix="index/index"):
    if kind == "faiss" and _HAS_FAISS:
        idx = faiss.read_index(path_prefix + ".faiss")
        return ("faiss", idx)
    else:
        arr = np.load(path_prefix + ".npy")
        return ("numpy", arr)

# -----------------------------
# Retrieval utilities
# -----------------------------
def faiss_search(index, query_vec: np.ndarray, top_k=8) -> List[Tuple[int,float]]:
    D, I = index.search(query_vec.reshape(1,-1), top_k)
    sims = D[0].tolist()
    ids = I[0].tolist()
    return list(zip(ids, sims))

def numpy_search(emb_matrix: np.ndarray, query_vec: np.ndarray, top_k=8) -> List[Tuple[int,float]]:
    # both normalized
    sims = (emb_matrix @ query_vec).reshape(-1)
    idxs = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in idxs]

def cosine_search(index_kind, index_obj, corpus_emb, q_vec, top_k=8):
    if index_kind == "faiss":
        return faiss_search(index_obj, q_vec, top_k)
    else:
        return numpy_search(index_obj, q_vec, top_k)

# BM25 helper
def build_bm25(corpus: List[str]):
    tokenized = [c.split() for c in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25

def bm25_rank(bm25, query: str, k=8):
    scores = bm25.get_scores(query.split())
    ranked = np.argsort(-scores)[:k].tolist()
    return ranked

# -----------------------------
# RRF fusion
# -----------------------------
def reciprocal_rank_fusion(rank_lists: List[List[int]], k: int = 60, top_k: int = 8) -> List[int]:
    scores = {}
    for lst in rank_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    best = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
    return [int(doc_id) for doc_id, _ in best]

# -----------------------------
# HyDE + query variants (local generator)
# -----------------------------
Q_VARIANTS_PROMPT = "Rewrite the user query into {n} short, diverse search queries (one per line). Query: {q}"
HYDE_PROMPT = "Write a concise factual paragraph that likely answers: \"{q}\". Use key terms and entities. 60-140 words."

def generate_text_local(gen, tok, prompt: str, device: str, max_new_tokens: int = 160, temperature: float = 0.0) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    with torch.no_grad():
        out = gen.generate(**inputs, **gen_kwargs)
    decoded = tok.batch_decode(out, skip_special_tokens=True)
    return decoded[0].strip()

def make_query_variants(gen, tok, device, q: str, n: int = 4) -> List[str]:
    prompt = Q_VARIANTS_PROMPT.format(q=q, n=n)
    txt = generate_text_local(gen, tok, prompt, device, max_new_tokens=120)
    lines = [l.strip(" -•\t") for l in txt.split("\n") if l.strip()]
    uniq = []
    seen = set()
    for l in lines + [q]:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    return uniq[:n]

def hyde_doc_local(gen, tok, device, q: str) -> str:
    prompt = HYDE_PROMPT.format(q=q)
    return generate_text_local(gen, tok, prompt, device, max_new_tokens=200)

# -----------------------------
# Answer generation (local)
# -----------------------------
ANSWER_PROMPT = """You are an assistant. Use ONLY the provided context and cite file:chunk inline like (file:chunk).
Question: {q}

Context:
{ctx}

Answer concisely in 2-5 sentences with inline citations.
"""

def answer_with_local_gen(gen, tok, device, q: str, context: str, max_new_tokens=200):
    prompt = ANSWER_PROMPT.format(q=q, ctx=context)
    return generate_text_local(gen, tok, device=device, prompt=prompt) if False else generate_text_local(gen, tok, prompt, device, max_new_tokens=max_new_tokens)

# -----------------------------
# Pipeline class
# -----------------------------
class HyDeRAG:
    def __init__(self, pdf_folder: str, index_prefix: str = "index/index", rebuild: bool = False,
                 chunk_size_words: int = 300, overlap_words: int = 50):
        self.pdf_folder = pdf_folder
        self.index_prefix = index_prefix
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.rebuild = rebuild

        # storage
        self.corpus: List[str] = []
        self.metas: List[Dict] = []

        # models (lazy load)
        self.embedder = None
        self.gen_tok = None
        self.gen = None
        self.device = None

        # index
        self.index_kind = None
        self.index_obj = None
        self.corpus_emb = None
        self.bm25 = None

        # try to load existing index/metadata
        if not rebuild and os.path.exists(index_prefix + ".npy") or ( _HAS_FAISS and os.path.exists(index_prefix + ".faiss")):
            try:
                self._load_store()
                print("[INFO] Loaded stored index & metadata.")
            except Exception as e:
                print(f"[WARN] Could not load stored index: {e}. Will rebuild if possible.")

    def build(self, embed_model="sentence-transformers/all-MiniLM-L6-v2", gen_model="google/flan-t5-base",
              use_faiss: bool = _HAS_FAISS, batch_size: int = 64):
        print("[INFO] Building corpus from PDFs...")
        corpus, metas = build_corpus_from_folder(self.pdf_folder, self.chunk_size_words, self.overlap_words)
        if not corpus:
            raise RuntimeError("No PDFs or no text extracted. Put PDFs in folder or check parser.")
        self.corpus = corpus
        self.metas = metas

        print(f"[INFO] Loading models (embedder={embed_model}, gen={gen_model})...")
        self.embedder, self.gen_tok, self.gen, self.device = load_models(device=None, embed_model=embed_model, gen_model=gen_model)

        print("[INFO] Building embeddings (this may take some seconds)...")
        emb = build_embeddings(self.embedder, self.corpus, batch_size=batch_size, normalize=True)
        self.corpus_emb = emb
        kind = "faiss" if use_faiss and _HAS_FAISS else "numpy"
        print(f"[INFO] Building index (kind={kind})...")
        self.index_kind, self.index_obj = build_index(emb)
        print("[INFO] Building BM25 lexical index...")
        self.bm25 = build_bm25(self.corpus)

        print("[INFO] Persisting index and metadata...")
        save_index(self.index_kind, self.index_obj, self.index_prefix)
        save_json(self.index_prefix + ".meta.json", {"metas": self.metas, "corpus_len": len(self.corpus), "corpus_preview": self.corpus[:8]})
        print("[INFO] Build complete.")

    def _load_store(self):
        # load metadata and index (numpy or faiss)
        meta = load_json(self.index_prefix + ".meta.json")
        self.metas = meta["metas"]
        # lazy load corpus: we will reconstruct corpus by reading PDFs (safer)
        self.corpus, _ = build_corpus_from_folder(self.pdf_folder, self.chunk_size_words, self.overlap_words)
        # load embeddings/index
        if _HAS_FAISS and os.path.exists(self.index_prefix + ".faiss"):
            self.index_kind, self.index_obj = load_index("faiss", self.index_prefix)
            # we also need embeddings for numpy fallback; try to load .npy if present
            if os.path.exists(self.index_prefix + ".npy"):
                self.corpus_emb = np.load(self.index_prefix + ".npy")
        elif os.path.exists(self.index_prefix + ".npy"):
            self.index_kind, self.index_obj = load_index("numpy", self.index_prefix)
            self.corpus_emb = self.index_obj
        else:
            raise RuntimeError("No index files found.")

        # build bm25
        self.bm25 = build_bm25(self.corpus)

    def ensure_models(self, embed_model="sentence-transformers/all-MiniLM-L6-v2", gen_model="google/flan-t5-base"):
        if self.embedder is None or self.gen is None:
            self.embedder, self.gen_tok, self.gen, self.device = load_models(device=None, embed_model=embed_model, gen_model=gen_model)
            if self.corpus_emb is None and os.path.exists(self.index_prefix + ".npy"):
                self.corpus_emb = np.load(self.index_prefix + ".npy")

    def retrieve(self, query: str, n_variants: int = 4, per_variant_k: int = 8, final_top_k: int = 6, rrf_k: int = 60, use_hyde: bool = True):
        self.ensure_models()
        # make variants
        variants = make_query_variants(self.gen, self.gen_tok, self.device, query, n=n_variants)
        rank_lists = []

        for v in variants:
            # HyDE: generate hypothetical passage for the variant (optional)
            if use_hyde:
                hypo = hyde_doc_local(self.gen, self.gen_tok, self.device, v)
                q_vec = self.embedder.encode([hypo], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
            else:
                q_vec = self.embedder.encode([v], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")

            # dense retrieval
            dense_hits = cosine_search(self.index_kind, self.index_obj, self.corpus_emb, q_vec, top_k=per_variant_k)
            dense_ids = [doc_id for doc_id, _ in dense_hits]
            rank_lists.append(dense_ids)

            # lexical retrieval (BM25)
            bm25_ids = bm25_rank(self.bm25, v, k=per_variant_k)
            rank_lists.append(bm25_ids)

        fused = reciprocal_rank_fusion(rank_lists, k=rrf_k, top_k=final_top_k)
        return fused, variants, rank_lists

    def answer(self, q: str, doc_ids: List[int], max_ctx_chars: int = 3500) -> str:
        # build short context with provenance
        parts = []
        total = 0
        for i in doc_ids:
            txt = self.corpus[i].strip()
            tag = f"({self.metas[i]['file']}:{self.metas[i]['chunk_id']})"
            part = f"{txt}\n{tag}"
            if total + len(part) > max_ctx_chars:
                break
            parts.append(part)
            total += len(part)
        context = "\n\n---\n\n".join(parts)
        # generate
        out = generate_text_local(self.gen, self.gen_tok, ANSWER_PROMPT.format(q=q, ctx=context), self.device, max_new_tokens=220)
        return out

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build","query"], required=True, help="build index or query")
    ap.add_argument("--pdf_folder", required=True, help="folder with pdfs")
    ap.add_argument("--index_prefix", default="index/index", help="prefix for index files")
    ap.add_argument("--query", default=None)
    ap.add_argument("--n_variants", type=int, default=4)
    ap.add_argument("--per_variant_k", type=int, default=8)
    ap.add_argument("--final_top_k", type=int, default=6)
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--use_faiss", action="store_true", help="prefer faiss if available")
    ap.add_argument("--use_hyde", action="store_true", help="enable HyDE")
    ap.add_argument("--show_sources", action="store_true", help="print top source list")
    args = ap.parse_args()

    rag = HyDeRAG(args.pdf_folder, index_prefix=args.index_prefix, rebuild=(args.mode=="build"))
    if args.mode == "build":
        rag.build(use_faiss=args.use_faiss)
        print("[DONE] index built.")
        return

    # query mode
    if args.mode == "query":
        if not args.query:
            print("Provide --query")
            return
        # ensure we have index (load or build)
        try:
            rag._load_store()
        except Exception:
            print("[INFO] No stored index; building now (this will take time)...")
            rag.build(use_faiss=args.use_faiss)
        rag.ensure_models()
        doc_ids, variants, rank_lists = rag.retrieve(args.query, n_variants=args.n_variants,
                                                     per_variant_k=args.per_variant_k,
                                                     final_top_k=args.final_top_k,
                                                     rrf_k=args.rrf_k,
                                                     use_hyde=args.use_hyde)
        answer = rag.answer(args.query, doc_ids)
        print("\n=== QUESTION ===\n", args.query)
        print("\n=== VARIANTS ===")
        for v in variants:
            print("- ", v)
        print("\n=== ANSWER ===\n")
        print(answer)
        if args.show_sources:
            print("\n=== TOP SOURCES ===")
            for i in doc_ids:
                m = rag.metas[i]
                print(f"- {m['file']} (chunk {m['chunk_id']})")
        return

if __name__ == "__main__":
    main()
