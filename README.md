An intelligent, AI-powered exam generation system that converts PDF files to Markdown and uses Google's Gemini API with embedding-based retrieval to create contextually relevant, comprehensive exam papers from educational content.

## 🎯 Key Features

- **📄 PDF to Markdown Conversion**: Automated conversion of PDF files to structured markdown format[^1][^2]
- **🧠 AI-Powered Content Processing**: Uses Google's text-embedding-004 model for semantic understanding[^3][^4]
- **📝 Comprehensive Exam Generation**: Creates complete academic exam papers with AI-generated model answers and detailed marking schemes[^2][^5]
- **🔍 RAG Pipeline**: Retrieval-Augmented Generation for contextually relevant question creation[^6][^5]
- **📊 Multiple Output Formats**: Supports TXT, JSON, Markdown, and PDF output formats[^5]
- **💾 Vector Storage**: Integration with Supabase for scalable vector database operations[^7]
- **⚡ Content Classification**: Automatically distinguishes between exam papers, model answers, and lecture notes[^8]
- **🎯 Academic Quality**: Generates university-level content with proper academic structure and formatting[^5]


## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key
- Supabase account (optional, for vector storage)


### Installation

```bash
git clone <repository-url>
cd gemini-embedding-model
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install NLP dependencies
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```


### Environment Setup

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
SUPABASE_ANON_KEY=your_supabase_anon_key
RATE_LIMIT_RPM=15
BATCH_SIZE=10
```


## 📁 Project Structure

```
gemini-embedding-model/
├── config/                    # Configuration settings
│   ├── settings.py            # Main configuration file
│   └── pdf_conversion_config.py
├── data/
│   ├── input/                 # Original PDF files
│   │   ├── kelvin_papers/     # Exam papers and model answers
│   │   └── lectures/          # Lecture note PDFs
│   └── output/
│       ├── converted_markdown/ # Converted .md files
│       ├── processed/         # Processed chunks and embeddings
│       ├── generated_exams/   # Generated exam papers
│       └── logs/             # Comprehensive logging
├── src/core/
│   ├── embedding/            # Gemini API integration
│   ├── generation/           # Exam and question generation
│   ├── storage/              # Vector storage and database
│   └── text/                 # Text processing and chunking
├── scripts/
│   └── direct_convert.py     # PDF to Markdown conversion
└── run_pipeline.py          # Main CLI interface
```


## 🔄 Complete Workflow Commands

### Step 1: Convert PDFs to Markdown

**Convert all PDFs in input directories:**

```bash
python scripts/direct_convert.py
```

This command will:

- Convert all PDFs in `data/input/kelvin_papers/` and `data/input/lectures/`
- Generate markdown files in `data/output/converted_markdown/`
- Maintain directory structure and file organization
- Create a comprehensive conversion log


### Step 2: Run Complete Pipeline

**Generate comprehensive exam papers (recommended):**

```bash
python run_pipeline.py run-full-pipeline
```

This single command executes the entire workflow:

1. **Text Processing**: Processes converted markdown files with content classification[^8]
2. **Embedding Generation**: Creates semantic embeddings using Gemini's text-embedding-004[^9][^10]
3. **Exam Generation**: Generates comprehensive exam papers with AI model answers and marking schemes[^5]

### Step 3: Individual Pipeline Commands

**Process converted markdown files:**

```bash
python run_pipeline.py process-texts --input-dir=data/output/converted_markdown
```

**Generate embeddings from processed content:**

```bash
python run_pipeline.py generate-embeddings --batch-size=10 --use-supabase
```

**Generate structured exam papers:**

```bash
python run_pipeline.py generate-structured-exam \
  --topic "AI and Data Analytics" \
  --structure-type standard \
  --formats txt,md,pdf,json
```


## 📋 Specialized Generation Commands

### Generate Different Types of Papers

**Generate comprehensive exam with model answers:**

```bash
python run_pipeline.py generate-structured-exam \
  --topic "Machine Learning Fundamentals" \
  --formats txt,md,pdf,json \
  --quota-aware
```

**Generate template-only exam (when API quota is exhausted):**

```bash
python run_pipeline.py generate-structured-exam \
  --topic "Data Science Applications" \
  --template-only \
  --formats txt,md,pdf
```

**Generate exam with specific difficulty:**

```bash
python run_pipeline.py generate-structured-exam \
  --topic "Neural Networks and Deep Learning" \
  --structure-type advanced \
  --formats pdf,json
```


## 🎯 Advanced Usage Options

### Content Processing Options

```bash
# Force reprocess existing files
python run_pipeline.py process-texts --force-reprocess

# Process without Supabase integration
python run_pipeline.py process-texts --use-supabase=false

# Generate embeddings with custom batch size
python run_pipeline.py generate-embeddings --batch-size=20 --force-regenerate
```


### Exam Generation Options

```bash
# Generate with specific output formats
python run_pipeline.py generate-structured-exam --formats md,pdf

# Generate using only local data (no Supabase)
python run_pipeline.py generate-structured-exam --use-supabase=false

# Generate with quota awareness enabled
python run_pipeline.py generate-structured-exam --quota-aware --template-only=false
```


## 📊 System Management Commands

### Check System Status

```bash
# View comprehensive system status
python run_pipeline.py status

# Test database connection and schema
python run_pipeline.py test-database

# Check for duplicate content
python run_pipeline.py check-duplicates
```


### View and Manage Logs

```bash
# View recent application logs
python run_pipeline.py logs --log-type application --lines 100

# View error logs
python run_pipeline.py logs --log-type errors --lines 50

# Follow logs in real-time
python run_pipeline.py logs --tail

# Check logging system status
python run_pipeline.py log-status

# Clear all log files (with confirmation)
python run_pipeline.py clear-logs
```


## 📄 Generated Output Files

### Exam Papers Generated

The system creates comprehensive academic exam papers in multiple formats:

- **Questions Only**: `comprehensive_ai_questions_YYYYMMDD_HHMMSS.{txt,md,pdf}`
- **Model Answers**: `comprehensive_ai_answers_YYYYMMDD_HHMMSS.{txt,md,pdf}`
- **Marking Schemes**: `comprehensive_ai_schemes_YYYYMMDD_HHMMSS.{txt,md,pdf}`
- **Complete Exam**: `comprehensive_ai_complete_exam_YYYYMMDD_HHMMSS.json`


### Sample Output Structure

```
data/output/generated_exams/
├── comprehensive_ai_questions_20250731_120000.txt     # Question paper
├── comprehensive_ai_questions_20250731_120000.md      # Markdown format
├── comprehensive_ai_questions_20250731_120000.pdf     # PDF format
├── comprehensive_ai_answers_20250731_120000.txt       # AI model answers
├── comprehensive_ai_answers_20250731_120000.md        # Markdown format
├── comprehensive_ai_answers_20250731_120000.pdf       # PDF format
├── comprehensive_ai_schemes_20250731_120000.txt       # Marking schemes
├── comprehensive_ai_schemes_20250731_120000.md        # Markdown format
├── comprehensive_ai_schemes_20250731_120000.pdf       # PDF format
└── comprehensive_ai_complete_exam_20250731_120000.json # Complete data
```


## 🔧 Configuration Options

### Key Settings in `config/settings.py`

```python
# API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
RATE_LIMIT_RPM = 15  # Requests per minute

# Processing Configuration
BATCH_SIZE = 10
MAX_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSIONS = 768

# Input/Output Directories
INPUT_DIR = DATA_DIR / "output" / "converted_markdown"  # Processes converted markdown
MARKDOWN_INPUT_DIR = DATA_DIR / "output" / "converted_markdown"
OUTPUT_DIR = DATA_DIR / "output"
```


## 🎓 Academic Features

### Exam Paper Structure

The system generates comprehensive academic exam papers with:

- **University-level questions** across multiple difficulty levels
- **Detailed model answers** with academic explanations
- **Comprehensive marking schemes** with point-by-point criteria
- **Proper academic formatting** with instructions and layouts
- **Content classification** distinguishing exam papers from lecture notes[^8]


### AI-Powered Content Generation

- **Contextual question creation** using RAG methodology[^6][^5]
- **Comprehensive model answers** with detailed explanations[^5]
- **Academic marking schemes** with criterion-based assessment[^5]
- **Content source tracking** for generated materials[^5]


## ⚡ Performance and Optimization

### API Quota Management

The system includes intelligent quota management[^5]:

- **Rate limiting** respects Gemini API limits (15 requests/minute default)
- **Quota-aware generation** with fallback to template-based generation
- **Comprehensive error handling** with automatic retries
- **Template fallbacks** when API limits are reached


### Processing Efficiency

- **Duplicate detection** prevents reprocessing existing content[^7]
- **Batch processing** for efficient embedding generation[^9]
- **Content chunking** with overlap for better context preservation[^11]
- **Comprehensive caching** reduces redundant API calls[^5]


## 🔍 Troubleshooting

### Common Issues and Solutions

**1. Missing API Key**

```bash
# Ensure your .env file contains:
GEMINI_API_KEY=your_actual_api_key_here
```

**2. No Converted Markdown Files**

```bash
# First convert PDFs to markdown:
python scripts/direct_convert.py

# Then run the pipeline:
python run_pipeline.py run-full-pipeline
```

**3. API Quota Exhausted**

```bash
# Use template-only generation:
python run_pipeline.py generate-structured-exam --template-only
```

**4. spaCy Model Missing**

```bash
# Install required language model:
python -m spacy download en_core_web_sm
```


### Testing Connections

```bash
# Test Gemini API connection
python scripts/setup/test_api_connection.py

# Test complete system status
python run_pipeline.py status

# Test database connection
python run_pipeline.py test-database
```


## 📚 Core Technologies

- **Google Gemini API**: text-embedding-004 for embeddings, gemini-2.0-flash-exp for generation[^9][^10][^5]
- **Supabase**: Vector database for scalable storage and similarity search[^7]
- **spaCy \& NLTK**: Advanced text processing and chunking[^11]
- **RAG Pipeline**: Retrieval-Augmented Generation for contextual content creation[^6]
- **Academic AI**: University-level content generation with proper formatting[^5]


## 🎯 Use Cases

- **Academic Institutions**: Generate comprehensive exam papers from lecture materials
- **Educational Content Creation**: Convert PDF textbooks into structured question banks
- **Assessment Development**: Create model answers and marking schemes automatically
- **Content Analysis**: Analyze and classify educational materials using AI embeddings
- **Automated Evaluation**: Generate consistent marking criteria for academic assessments

This comprehensive system transforms static PDF educational content into dynamic, AI-powered exam generation capabilities, making it an essential tool for modern educational institutions and content creators.