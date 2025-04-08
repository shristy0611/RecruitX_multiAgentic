# RecruitX Project Roadmap

**Goal:** Develop a multi-agent web application that analyzes Job Descriptions (JDs) and Candidate CVs using Gemini AI to generate a sophisticated scoring system, identifying the best-fit candidates. Focus on creating a powerful AI assistant for recruiters.

**Core Principles:**

1.  **SOTA Driven:** Continuously research and integrate the best available techniques for each component (parsing, analysis, scoring, RAG).
2.  **Multi-Agent Architecture:** Design distinct AI agents with specific roles, leveraging Gemini's capabilities.
3.  **Scalability & Maintainability:** Build clean, well-structured code suitable for different environments (dev, test, prod).
4.  **User-Centric:** Focus on providing clear, actionable insights and explainability to recruiters.

**Current AI Agents (v1 Implemented & Tested):**

*   **JD Analysis Agent:** Extracts structured data from JDs via function calling.
*   **CV Analysis Agent:** Extracts structured data from CVs via JSON generation.
*   **Orchestration Agent (Scoring):** Performs 2-step scoring (skill extraction + synthesis).

**(Potential Later) RAG Agent:** Specialized agent for retrieval-augmented generation.

**High-Level Technology Stack:**

*   **Backend:** Python (FastAPI) **[Selected & Setup]**
*   **Frontend:** React + TypeScript + Vite **[Setup Initiated]**
*   **Database:** PostgreSQL + SQLAlchemy + Alembic **[Setup Complete]**
*   **AI:** Google Gemini API (`google-generativeai`, `gemini-2.0-flash-lite`, `text-embedding-004`) **[Integrated & Tested]**
*   **Vector Store:** ChromaDB **[Setup Initiated]**
*   **Deployment:** Docker, Cloud Platform (TBD)

---

## Roadmap Status

*Legend: ✅ Done | 🚧 In Progress / Backend Done | ⚪ Not Started*

**Phase 1: Foundation & Setup (Est. 1-2 Weeks)** 
`[ Estimated Completion: 100% ]`

1.  ✅ Project Initialization (Git, `README.md`, `ROADMAP.md`)
2.  ✅ Environment Setup (`.env`, `requirements.txt`)
3.  ✅ Backend Framework Setup (FastAPI)
4.  ✅ Basic API Endpoint (`/ping`)
5.  ✅ Gemini Connection Test (Script, API Key, Model Config)

**Phase 2: Core Data Handling & JD Agent (Est. 3-4 Weeks)** 
`[ Estimated Completion: 100% ]`

1.  ✅ Database Setup (Models, Schemas, Migrations)
2.  ✅ JD File Upload API (`/api/v1/jobs/upload`)
3.  ✅ JD Parsing Utility (`extract_text_from_file`)
4.  ✅ JD Analysis Agent v1 (Gemini Integration, Function Calling, DB Storage)
5.  ✅ JD Analysis Trigger API (`/api/v1/jobs/{job_id}/analyze`)

**Phase 3: CV Agent & Basic Matching (Est. 3-4 Weeks)** 
`[ Estimated Completion: 100% ]`

1.  ✅ CV Upload API (`/api/v1/candidates/upload`)
2.  ✅ CV Parsing (Utilizes `extract_text_from_file`)
3.  ✅ CV Analysis Agent v1 (Gemini Integration, JSON Output, DB Storage)
4.  ✅ Scoring Agent v1 (OrchestrationAgent: 2-step, Gemini, DB Storage)
5.  ✅ Scoring Trigger & Retrieval APIs (`/api/v1/scores/...`)
6.  ✅ Basic Candidate CRUD & Analysis APIs (`/api/v1/candidates/...`)

**Phase 4: Frontend Development & Integration (Est. 4-6 Weeks)** 
`[ Estimated Completion: 100% ]`

1.  ✅ Frontend Setup (React, TypeScript, Vite, Tailwind CSS, Routing)
2.  ✅ UI Components (Uploads, Job List, Candidate List, Job Details, Candidate Details done)
3.  ✅ API Integration (Uploads, Job List/Details, Candidate List/Details connected)
4.  ⚪ Professional Report Styling (UI/PDF Generation)

**Phase 5: SOTA Refinement & Advanced Features (Ongoing)** 
`[ Estimated Completion: 95% ]`

1.  **Agent Improvement:**
    *   ✅ Refine Prompts (Scoring done, JD refined)
    *   ✅ Convert CV Agent to use Function Calling (Completed)
    *   ✅ Improve Parsing Robustness (different file types/layouts)
2.  **RAG Implementation:**
    *   ✅ Setup Vector Store (`ChromaDB`, Embedding Function - `text-embedding-004`)
    *   ✅ Indexing Logic (Chunking with overlap, adding JDs/CVs to store via Services)
    *   🚧 Modify Agents to Retrieve Context (Scoring Agent done, Prompt Augmentation pending)
    *   ⚪ Add RAG for Grounded Market Insights?
    *   🚧 **Evolve to Agentic RAG:** (In Progress)
        *   ✅ Implement Query Analysis/Decomposition (JobRequirementFacet model, decompose_job_description method)
        *   ✅ Implement Dynamic Retrieval Strategies (AgenticRAGService, per-facet evidence retrieval)
        *   ✅ Implement Relevance/Quality Checks (Embedding cross-check validation)
        *   ✅ Implement Iterative Refinement Loops (Facet-specific query reformulation for required facets)
        *   ✅ Implement Tool Integration (External market data & salary benchmarking APIs)
3.  **Advanced Scoring:**
    *   ✅ Research/Implement Semantic Similarity Scoring (Complete: added document embedding generation, cosine similarity calculation, and integrated into scoring flow)
4.  **Recruiter-Focused UX Enhancements:**
    *   ✅ Advanced Candidate Ranking/Filtering (Backend API done)
    *   ✅ Improved Batch Processing (Backend concurrency done)
    *   ✅ Side-by-Side Comparison UI (Added job-specific candidate comparison page with skills, experience, education, and score comparison)
    *   ⚪ Interview Question Generation
    *   ⚪ Internal Notes/Collaboration Features
    *   ⚪ CV Highlighting Feature
5.  **Testing:**
    *   🚧 Implement comprehensive Unit & Integration Tests (In Progress - 70% coverage overall)
        * ✅ `simple_scoring_agent.py` tests complete
        * ✅ `candidate_service.py` tests complete
        * ✅ `job_service.py` tests complete
        * ✅ `agentic_rag_service.py` tests complete (91% coverage, improved from 50%)
        * ✅ `jd_analysis_agent.py` tests (95% coverage)
        * ✅ `code_execution_agent.py` tests complete (97% coverage)
        * ✅ Tests for `multimodal_agent.py` (complete)
        * ✅ Tests for `integrated_agent.py` (95% coverage)
        * ✅ Tests for `tool_use_agent.py` (92% coverage)
6.  **Deployment Strategy:**
    *   ⚪ Define CI/CD Pipelines
    *   ⚪ Choose Cloud Platform & Setup
7.  **Future Considerations:**
    *   ⚪ ATS Integration Planning 

## Testing Status (Current Coverage: 70%)

- ✅ Comprehensive unit tests for `simple_scoring_agent.py` (77% coverage)
- ✅ Unit tests for `candidate_service.py` (97% coverage) 
- ✅ Unit tests for `job_service.py` (89% coverage)
- ✅ Unit tests for `cv_analysis_agent.py` (84% coverage)
- ✅ Unit tests for `agentic_rag_service.py` (91% coverage)
- ✅ Unit tests for `jd_analysis_agent.py` (95% coverage)
- ✅ Unit tests for `vector_db_service.py` (70% coverage)
- ❌ Integration tests for API endpoints (50% coverage)
- ✅ Tests for `integrated_agent.py` (95% coverage)
- ✅ Tests for `multimodal_agent.py` (complete)
- ✅ Tests for `tool_use_agent.py` (92% coverage) 