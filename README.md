# RAGASuite - RAG Evaluation Dashboard

RAGASuite is a Streamlit-based application to upload documents, run RAG pipelines, generate datasets, and evaluate RAG metrics such as context precision, context relevancy, and answer relevancy.

## Features

- Upload PDFs for document ingestion
- Select your LLM and RAG pipeline (Simple RAG or MultiQuery RAG)
- Generate synthetic or manual datasets
- Auto-generate answers using your selected RAG pipeline
- Evaluate with multiple metrics to assess pipeline performance

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/fran-gen/ragasuite.git
cd ragasuite
```

### 2. Install dependencies with Poetry
```bash
poetry install
```

### 3. Activate Poetry shell
```bash
poetry shell
```

### 4. Set environment variables
Export your environment variables in your shell:
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

Or create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the App

To start the Streamlit app:
```bash
poetry run streamlit run app.py
```

## Usage Flow

### Step 1: Upload PDF
- Upload your document (PDF format).

### Step 2: Select RAG pipeline
- Choose an LLM (e.g., `gpt-4o`).
- Select between `Simple RAG` or `MultiQuery RAG`.

### Step 3: Generate Dataset
- Choose between synthetic or manual dataset generation.
- Click "Generate Dataset".

### Step 4: Generate Answer
- Click "Generate answer" to let the pipeline produce responses.

### Step 5: Evaluation
- Select evaluation metrics: context precision, context relevancy, and answer relevancy.
- Click "Generate Evaluation" to get evaluation results in a table format.

## Notes

- The app currently supports PDF input with a configurable file size limit.
- It is modular and can be extended with additional pipelines or custom metrics.
