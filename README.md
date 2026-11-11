# AI Paper Assistant

**A lightweight Retrieval-Augmented Generation (RAG) system built from scratch using Python, Sentence Transformers, and the DeepSeek API.**

This project explores how large language models can answer questions about academic papers by retrieving relevant information from documents.

Rather than relying on an end-to-end RAG framework, the core pipeline is implemented manually to understand how document processing, semantic retrieval, and LLM-based answer generation work together.

## 1. Project Overview

Large language models are powerful tools for understanding research papers, but they do not always have access to the specific documents a user wants to analyze.

Retrieval-Augmented Generation (RAG) addresses this problem by retrieving relevant external information and providing it to an LLM as context.

This project implements a minimal RAG pipeline for academic paper question answering.

**Main objectives:**

- Understand the architecture of a RAG system.
- Extract and preprocess text from PDF documents.
- Explore text chunking and semantic embeddings.
- Implement vector similarity search.
- Generate answers grounded in retrieved context.
- Evaluate retrieval performance and identify limitations.

The system was developed and tested using the paper *Attention Is All You Need* (Vaswani et al., 2017).

## 2. System Architecture

```text
                 Research Paper (PDF)
                          |
                          v
                 PDF Text Extraction
                       PyMuPDF
                          |
                          v
                    Text Chunking
               Chunk Size / Overlap
                          |
                          v
                   Text Embeddings
                 all-MiniLM-L6-v2
                          |
                          v
                   Chunk Embeddings
                     (N × 384)
                          |
                          |
User Question             |
      |                   |
      v                   |
Query Embedding           |
      |                   |
      +-------------------+
                |
                v
        Cosine Similarity
                |
                v
          Top-k Retrieval
                |
                v
     Retrieved Context + Question
                |
                v
           DeepSeek LLM
                |
                v
         Generated Answer
```

The architecture separates **information retrieval** from **answer generation**.

The embedding model identifies potentially relevant text chunks, while the LLM uses the retrieved original text to generate a response.

## 3. Implementation

### 3.1 PDF Processing

PDF documents are processed using PyMuPDF.

Text is extracted page by page and combined into a single document for further processing.

### 3.2 Text Chunking

The extracted text is divided into overlapping chunks.

The implementation uses a basic sentence-aware splitting strategy, attempting to preserve sentence boundaries where possible.

Default parameters:

- Chunk size: 1,000 characters
- Overlap: 200 characters

For the selected research paper, this produced 54 text chunks.

### 3.3 Semantic Embeddings

The project uses the Sentence Transformers model:

`sentence-transformers/all-MiniLM-L6-v2`

Each chunk is represented by a 384-dimensional embedding vector.

The same embedding model converts user questions into vectors for semantic similarity comparison.

### 3.4 Retrieval

Cosine similarity is used to compare the query embedding with document chunk embeddings.

Chunks are ranked by similarity score, and the top-k results are selected as supporting context.

The default RAG experiment uses `k = 3`.

### 3.5 Answer Generation

Retrieved chunks are combined with the user's question to construct a context-grounded prompt.

The DeepSeek API is then used to generate an answer.

The prompt instructs the model to rely on the supplied context and acknowledge when sufficient evidence is unavailable.

## 4. Project Structure

```text
ai-paper-assistant/
├── main.py                 # Basic DeepSeek API experiment
├── parse_pdf.py            # PDF text extraction
├── chunk_text.py           # Text chunking
├── embedding_test.py       # Embedding and similarity experiment
├── retrieve.py             # Semantic retrieval
├── rag.py                  # RAG question answering
├── eval_questions.py       # Evaluation questions and reference evidence
├── evaluate_retrieval.py   # Retrieval evaluation
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

Generated files such as `output.txt`, local PDF files, and environment variables are excluded from version control.

## 5. Installation and Usage

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-paper-assistant.git
cd ai-paper-assistant
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Install dependencies

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

### Step 3: Configure the API key

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_api_key_here
```

Do not commit this file to GitHub.

### Step 4: Prepare the research paper

Download the PDF version of:

**Attention Is All You Need**

https://arxiv.org/abs/1706.03762

Save the PDF as `paper.pdf` in the project root.

### Step 5: Extract document text

```bash
python parse_pdf.py
```

The extracted text is saved to `output.txt`.

### Step 6: Run semantic retrieval

```bash
python retrieve.py
```

The script ranks text chunks by similarity to the query and displays the retrieved results.

The query is currently defined in `retrieve.py` and can be changed directly.

### Step 7: Run RAG question answering

```bash
python rag.py
```

The script retrieves relevant chunks, constructs a prompt, and requests an answer from DeepSeek.

### Step 8: Evaluate retrieval performance

```bash
python eval_questions.py
python evaluate_retrieval.py
```

The evaluation script measures evidence retrieval using Hit@1, Hit@3, and Hit@5.

## 6. Preliminary Evaluation

A small evaluation dataset containing five questions was manually constructed from *Attention Is All You Need*.

The questions cover the optimizer, attention heads, encoder layers, training data, and translation performance.

### Results

| Metric | Result |
|---|---|
| Hit@1 | 40.0% (2/5) |
| Hit@3 | 80.0% (4/5) |
| Hit@5 | 80.0% (4/5) |

**Interpretation:**

Increasing the retrieval depth from Top-1 to Top-3 improved evidence retrieval on this small evaluation set.

Increasing it further to Top-5 did not produce additional hits.

For the question about the number of attention heads, the annotated evidence first appeared at rank 9, revealing a potential limitation of the retrieval pipeline.

### Evaluation Limitations

The evaluation uses phrase matching against manually selected evidence, rather than full semantic relevance judgments.

Consequently:

- A correct passage may be missed if its wording differs from the reference phrase.
- A phrase match does not necessarily guarantee sufficient evidence.
- The five-question sample is too small for broad performance claims.
- Retrieval success does not guarantee that the generated answer is correct.

These results should therefore be interpreted as preliminary evidence-retrieval measurements rather than comprehensive RAG quality benchmarks.

## 7. Key Learnings

This project provided hands-on experience with several fundamental concepts in modern LLM applications:

**RAG architecture:** Understanding the separation between document retrieval and language generation.

**Semantic representation:** Exploring how embedding models represent text as numerical vectors and enable similarity-based search.

**System design:** Understanding how document parsing, chunking, embedding, retrieval, and prompt construction interact.

**Evaluation:** Learning why a working AI system is not necessarily a reliable one, and how retrieval metrics and error analysis can support systematic improvement.

**Engineering trade-offs:** Recognizing the relationship between context size, token consumption, retrieval quality, and implementation complexity.

## 8. Limitations and Future Work

This is an educational prototype rather than a production-ready research assistant.

Current limitations include:

- Evaluation on a single research paper.
- A small manually constructed evaluation dataset.
- Basic text extraction without comprehensive figure, formula, or table understanding.
- A single embedding model with cosine similarity retrieval.
- No hybrid search or reranking.
- No systematic comparison with full-context prompting.
- No autonomous research Agent or multi-step reasoning workflow.

Potential future extensions include:

- Comparing RAG with direct long-context prompting.
- Exploring hybrid retrieval and reranking.
- Using modern document parsing and retrieval frameworks.
- Supporting multiple documents and cross-paper analysis.
- Developing tool-using research agents.

These extensions are not part of the current implementation.

## 9. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| PDF processing | PyMuPDF |
| Embedding model | all-MiniLM-L6-v2 |
| Embedding library | Sentence Transformers |
| Similarity computation | NumPy / scikit-learn |
| LLM | DeepSeek API |
| API integration | OpenAI Python SDK |
| Evaluation | Custom Hit@k implementation |

## 10. Project Status

**Completed — Educational Prototype (V0–V5)**

The project has reached its intended learning milestone: building a minimal, explainable RAG system and conducting an initial retrieval evaluation.

It serves as a foundation for further exploration of LLM applications, NLP, information retrieval, and agent-based systems.

## References

- Vaswani et al. (2017), *Attention Is All You Need*: https://arxiv.org/abs/1706.03762
- Sentence Transformers: https://www.sbert.net/
- PyMuPDF: https://pymupdf.readthedocs.io/
- DeepSeek API: https://api-docs.deepseek.com/