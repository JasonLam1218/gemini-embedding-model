# Gemini Embedding Model - Exam Generator

Generate AI-powered exam questions from educational content using Google's Gemini AI and vector search.

## 🎯 What It Does

- Takes manually copied text from exam papers/lectures
- Creates semantic embeddings using Gemini AI
- Stores vectors in Supabase database
- Generates new exam questions using RAG (Retrieval-Augmented Generation)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and setup
git clone 
cd gemini-embedding-model
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_KEY=your_supabase_service_key
```

### 3. Setup Database

```bash
python scripts/setup/test_api_connection.py
python scripts/setup/initialize_database.py
```

## 📝 Usage

### Add Your Content

Put text files in `data/input/kelvin_papers/set_1/`:

**paper_1_text.txt**:
```txt
Section A: Multiple Choice

1. What is machine learning?
a) A type of computer
b) AI technique for learning patterns
c) A programming language
d) A database system

2. Which algorithm is supervised learning?
a) K-means
b) Linear regression  
c) PCA
d) Clustering
```

**paper_1_metadata.json**:
```json
{
  "title": "AI Fundamentals Exam",
  "topics": ["machine learning", "algorithms"],
  "difficulty": "intermediate"
}
```

### Generate Exam

```bash
# Full pipeline
python run_pipeline.py run-full-pipeline

# Or step by step:
python run_pipeline.py process-texts
python run_pipeline.py generate-embeddings  
python run_pipeline.py generate-exam --topic "machine learning" --num-questions 5
```

## 📁 Project Structure

```
gemini-embedding-model/
├── config/
│   └── settings.py              # Configuration
├── data/
│   ├── input/kelvin_papers/     # Your text files go here
│   └── output/logs/             # Generated results
├── src/core/
│   ├── embedding/               # Gemini API integration
│   ├── text/                    # Text processing
│   ├── storage/                 # Supabase database
│   └── generation/              # Exam generation
├── scripts/setup/               # Setup utilities
├── requirements.txt
└── run_pipeline.py             # Main interface
```

## 🔧 Common Commands

```bash
# Test connections
python scripts/setup/test_api_connection.py

# Process new text files
python run_pipeline.py process-texts --input-dir data/input/kelvin_papers

# Generate specific exam
python run_pipeline.py generate-exam --topic "neural networks" --num-questions 10 --difficulty "advanced"
```

## ⚠️ Troubleshooting

**Rate Limiting Error?**
- Reduce `RATE_LIMIT_RPM=30` in `.env`

**Database Connection Failed?**
- Check your Supabase URL and keys in `.env`
- Run `python scripts/setup/test_api_connection.py`

**Out of Memory?**
- Reduce batch size: set `BATCH_SIZE=5` in `.env`

**No Questions Generated?**
- Check if embeddings were created: look in `data/output/processed/`
- Try broader search: lower similarity threshold

## 🛠️ Requirements

- Python 3.9+
- Gemini API key from Google AI Studio
- Supabase account with vector extension enabled

## 📚 Key Files to Know

- `run_pipeline.py` - Main command interface
- `config/settings.py` - All configuration settings
- `src/core/embedding/gemini_client.py` - Gemini API wrapper
- `src/core/generation/exam_generator.py` - Question generation logic

That's it! Start by adding your text files and running the pipeline. Check the logs in `data/output/logs/` if you run into issues.