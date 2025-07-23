# Gemini Embedding-Based Exam Generation System

An intelligent exam generation system that uses Google's Gemini API and embedding-based retrieval to create contextually relevant exam questions from educational content.

## Features

- **Automated Text Processing**: Converts educational documents into optimized chunks for embedding generation[^1]
- **Semantic Search**: Uses Google's text-embedding-004 model for high-quality vector representations[^2]
- **RAG Pipeline**: Retrieval-Augmented Generation for contextually relevant question creation[^3]
- **Multiple Question Types**: Supports multiple-choice and short-answer questions[^4]
- **Structured Exam Papers**: Generates complete, formatted exam papers with proper academic structure[^4]
- **Vector Storage**: Integration with Supabase for scalable vector database operations[^5]
- **CLI Interface**: Easy-to-use command-line interface for all operations[^1]


## Installation

### Prerequisites

- Python 3.8+
- Google Gemini API key
- Supabase account (optional, for vector storage)


### Setup

1. **Clone and setup environment**:

```bash
git clone <repository-url>
cd gemini-embedding-model
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Install NLP dependencies**:

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

3. **Environment configuration**:
Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```


## Configuration

The system uses `config/settings.py` for configuration. Key settings include[^6]:

- `BATCH_SIZE`: Number of texts to process in each batch
- `RATE_LIMIT_RPM`: API rate limiting (requests per minute)
- Embedding and chunking parameters


## Usage

### Quick Start - Full Pipeline

Run the complete exam generation pipeline:

```bash
python run_pipeline.py run-full-pipeline
```

This will:

1. Process text files from `data/input/kelvin_papers/`
2. Generate embeddings using Gemini API
3. Create a structured exam paper

### Step-by-Step Usage

#### 1. Process Text Documents

```bash
python run_pipeline.py process-texts --input-dir data/input/kelvin_papers
```

Place your text files in the directory structure:

```
data/input/kelvin_papers/
├── set_1/
│   └── paper_1_text.txt
└── set_2/
    └── paper_2_text.txt
```


#### 2. Generate Embeddings

```bash
python run_pipeline.py generate-embeddings
```

Creates vector embeddings using Google's text-embedding-004 model[^2][^7].

#### 3. Generate Structured Exam

```bash
python run_pipeline.py generate-structured-exam --topic "AI and Data Analytics" --structure-type standard
```


### Check System Status

```bash
python run_pipeline.py status
```

Shows processing status, embedding counts, and generated exams.

## Project Structure

```
gemini-embedding-model/
├── config/                    # Configuration settings
├── data/
│   ├── input/                # Input text documents
│   └── output/               # Generated embeddings and exams
├── src/core/
│   ├── embedding/            # Gemini API integration and embedding generation
│   ├── generation/           # Exam and question generation
│   ├── storage/              # Vector storage and database operations
│   └── text/                 # Text processing and chunking
├── scripts/setup/            # Setup and testing utilities
└── run_pipeline.py          # Main CLI interface
```


### Core Components

| Component | Purpose |
| :-- | :-- |
| **GeminiClient**[^7] | Google Gemini API integration for embeddings and generation |
| **EmbeddingGenerator**[^2] | Manages embedding creation and similarity calculations |
| **TextChunker**[^8] | Intelligent text segmentation with overlap handling |
| **ExamGenerator**[^4] | Creates exam questions using RAG methodology |
| **VectorStore**[^5] | Supabase integration for vector storage and retrieval |
| **RAGPipeline**[^3] | Orchestrates retrieval and generation processes |

## Text Processing Pipeline

The system processes documents through several stages[^1][^8][^9]:

1. **Document Loading**: Extracts content from text files with metadata support
2. **Text Chunking**: Splits documents into overlapping chunks (1500 chars max, 200 char overlap)
3. **Embedding Generation**: Creates vector representations using Gemini's text-embedding-004
4. **Storage**: Saves embeddings with metadata for retrieval

## Exam Generation Process

The RAG-based exam generation follows this workflow[^4][^3]:

1. **Topic Analysis**: Generates query embeddings for the exam topic
2. **Content Retrieval**: Finds relevant chunks using cosine similarity
3. **Question Generation**: Uses Gemini's generative model to create questions
4. **Quality Validation**: Ensures questions meet formatting and content standards
5. **Paper Assembly**: Organizes questions into structured exam format

## API Requirements

### Google Gemini API

- **Embedding Model**: `text-embedding-004` for vector generation[^2][^7]
- **Generation Model**: `gemini-2.0-flash-exp` for question creation[^7][^4]
- **Rate Limiting**: Configured to respect API limits[^6]


### Supabase Integration

Optional vector database for production deployments[^5]:

- Document storage with metadata
- Embedding storage with similarity search
- Generated exam archival


## Output Examples

### Generated Files

- `data/output/processed/embeddings.json`: Vector embeddings with metadata
- `data/output/generated_exams/structured_exam_*.json`: Complete exam data
- `data/output/generated_exams/structured_exam_*.txt`: Formatted exam paper


### Exam Structure

Generated exams include:

- **Section A**: Multiple choice questions with 4 options each
- **Section B**: Short answer questions with mark allocations
- **Academic formatting**: Headers, instructions, and proper layout


## Error Handling and Logging

The system includes comprehensive error handling[^1][^2][^7]:

- **Rate limiting**: Automatic retry with exponential backoff
- **Content validation**: Ensures text quality before processing
- **Fallback mechanisms**: Alternative question generation methods
- **Detailed logging**: Progress tracking and error reporting


## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `GEMINI_API_KEY` is set in your `.env` file
2. **spaCy Model**: Run `python -m spacy download en_core_web_sm` if text processing fails
3. **Empty Embeddings**: Check input text files contain substantial content (>50 characters)
4. **Rate Limits**: The system automatically handles API rate limiting with retries

### Testing API Connection

```bash
python scripts/setup/test_api_connection.py
```