# HyDE + RAG-Fusion over Local PDFs

This project implements a **minimal but powerful Retrieval-Augmented Generation (RAG) workflow** using HyDE and Reciprocal Rank Fusion (RRF) for local PDF documents. It retrieves relevant chunks and generates concise, context-grounded answers with clear source citations.

## Features

- Ingest and chunk local PDFs efficiently
- Use a small, fast embedding model for retrieval (`all-MiniLM-L6-v2`)
- Apply HyDE to improve query relevance
- Fuse multiple query variants with Reciprocal Rank Fusion for higher accuracy
- Generate concise answers with inline source citations


## **Usage**

1. **Build the index**  
   Create embeddings and indices from your PDFs:  
   ```bash
   python test_Hyde.py --mode build --pdf_folder pdfs --use_faiss
   python test_Hyde.py --mode build --pdf_folder pdfs_dt --use_faiss

2. **Query the PDFs**
Run a query against the indexed PDFs:
```bash
python test_Hyde.py --mode query --pdf_folder pdfs --query "Give me details about BMW" --use_hyde --show_sources
python test_Hyde.py --mode query --pdf_folder pdfs_dt --query "Give me details about DT" --use_hyde --show_sources   

## Requirements

Python 3.10+ , Virtual environment recommended and the following packages:

```bash
pip install torch transformers sentence-transformers pymupdf scikit-learn numpy rank_bm25



