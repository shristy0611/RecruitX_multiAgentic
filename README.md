# RecruitX

A modern recruitment platform that helps match candidates with jobs using AI.

## Setup

### Backend Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Copy the `.env.example` to `.env` (if not already done) and update values as needed.

### Frontend Setup

1. Install frontend dependencies:
   ```bash
   cd recruitx_frontend
   npm install
   ```

## Running the Application

### Option 1: Run Frontend and Backend Separately

1. Run the backend:
   ```bash
   python run.py
   ```
   The backend will be available at http://localhost:8000

2. In a separate terminal, run the frontend:
   ```bash
   cd recruitx_frontend
   npm run dev
   ```
   The frontend will be available at http://localhost:3000

### Option 2: Run Both Together

Install the root-level dependencies:
```bash
npm install
```

Run both applications with one command:
```bash
npm run dev:all
```

## API Documentation

Once the backend is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

- Backend: FastAPI with Python
- Frontend: React with TypeScript, Vite, and TailwindCSS
- Database: SQLite by default (configurable in `.env`)

## Overview

RecruitX is an AI-powered recruitment platform that analyzes Job Descriptions (JDs) and Candidate CVs to generate sophisticated matching scores, identifying the best-fit candidates for specific roles. The system leverages Gemini 2.5 Pro's advanced capabilities, including massive context windows and multimodal understanding, to provide superior analysis and matching.

## Key Features

- **JD Analysis:** Upload and analyze job descriptions to extract structured information like required skills, experience, education, etc.
- **CV Analysis:** (Coming soon) Extract candidate information from resumes in various formats.
- **Matching & Scoring:** Generate detailed match scores with explanations using multiple AI capabilities.
- **Multi-Agent Architecture:** Specialized AI agents for each task using Google's Gemini 2.5 Pro.
- **API-First Design:** RESTful API endpoints for easy integration with frontends.

## Advanced Gemini 2.5 Pro Capabilities

RecruitX leverages the full spectrum of Gemini 2.5 Pro's capabilities:

### Function Calling & Structured Outputs
- Structured JD analysis with consistent JSON schema
- Well-defined output formats for all analyses
- Function calling for database interactions

### Search Grounding
- Market insights grounded in factual data
- Skill demand analysis based on current market trends
- Salary recommendations based on real-world data

### Code Execution
- Dynamic generation and execution of custom matching algorithms
- Skill similarity measurement using NLP techniques
- Visualization generation for match results

### Native Tool Use
- Database querying for job and candidate information
- Learning resource recommendations via API integration
- Skill database verification

### Thinking Process Transparency
- Explainable AI with detailed reasoning process
- Step-by-step analysis for match assessments
- Confidence scores for recommendations

### Multimodal Understanding (Coming Soon)
- Resume analysis including charts and visual elements
- OCR capabilities for document processing
- Comprehensive document understanding

## Technical Stack

- **Backend:** Python with FastAPI
- **Database:** SQLite (for development), PostgreSQL (for production)
- **AI Models:** Google Gemini 2.5 Pro (experimental-03-25)
- **File Handling:** Support for PDF, DOCX, and TXT formats

## Setup and Installation

### Prerequisites

- Python 3.9+ 
- pip
- Virtual environment tool (recommended)
- Google API key with access to Gemini 2.5 Pro

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/RecruitX.git
   cd RecruitX
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your `.env` file with your Gemini API keys:
   ```
   GEMINI_API_KEY_1=your_api_key_here
   GEMINI_API_KEY_2=another_api_key_here
   # ... add more as needed
   GEMINI_PRO_MODEL=gemini-2.5-pro-exp-03-25
   ```

### Database Initialization

Initialize the database with Alembic:

```bash
alembic upgrade head
```

### Running the Application

Start the development server:

```bash
python run.py
```

The API will be available at http://localhost:8000

## API Documentation

Once the server is running, you can access the interactive API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Job Description Management

- `GET /api/v1/jobs/` - List all jobs
- `GET /api/v1/jobs/{job_id}` - Get a specific job
- `POST /api/v1/jobs/` - Create a job manually
- `POST /api/v1/jobs/upload` - Upload a job description file (PDF, DOCX, TXT)
- `POST /api/v1/jobs/{job_id}/analyze` - Analyze a job description with Gemini 2.5 Pro

### Scoring & Matching

- `POST /api/v1/scores/` - Generate a match score between a job and a candidate
- `GET /api/v1/scores/{score_id}` - Get a specific score
- `GET /api/v1/scores/job/{job_id}` - Get all scores for a specific job
- `GET /api/v1/scores/candidate/{candidate_id}` - Get all scores for a specific candidate
- `POST /api/v1/scores/batch` - Generate scores for a job against multiple candidates

## Testing

RecruitX includes comprehensive test coverage for core components:

### Current Test Coverage: 62%

- Unit tests for all service layers
- Specialized tests for AI agents and their capabilities
- Mocked AI responses to enable reliable testing
- Integration tests for the scoring pipeline

Run the tests with coverage reporting:

```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=recruitx_app tests/

# Run specific test file
python -m pytest tests/unit/services/test_job_service.py -v
```

See `NEXT_STEPS.md` for the current testing roadmap and priorities.

## Implementation Roadmap

### Phase 1: Implemented ✅
- JD Analysis with Search Grounding
- Thinking capabilities for explainable results

### Phase 2: Implemented ✅
- Code Execution for dynamic matching algorithms 
- Native Tool Use for database and API integration

### Phase 3: Implemented ✅
- Integrated Multi-Capability Agent
- Comprehensive scoring service

### Phase 4: Coming Soon
- Frontend UI development
- Containerization and cloud deployment
- Integration with external recruitment platforms 