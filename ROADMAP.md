**Project:** RecruitX

**Goal:** Develop a multi-agent web application that analyzes Job Descriptions (JDs) and Candidate CVs using Gemini AI to generate a sophisticated scoring system, identifying the best-fit candidates. Focus on creating a powerful AI assistant for recruiters.

**Core Principles:**

1.  **SOTA Driven:** Continuously research and integrate the best available techniques for each component (parsing, analysis, scoring, RAG).
2.  **Multi-Agent Architecture:** Design distinct AI agents with specific roles, leveraging Gemini's capabilities.
3.  **Scalability & Maintainability:** Build clean, well-structured code suitable for different environments (dev, test, prod).
4.  **User-Centric:** Focus on providing clear, actionable insights and explainability to recruiters.

**Current AI Agents (Implemented & Tested):**

1.  **JD Analysis Agent:**
    *   Parses uploaded JDs (text extraction working).
    *   Uses Gemini to extract key requirements, skills, experience, etc., via function calling.
    *   Structures this information for comparison.
2.  **CV Analysis Agent:**
    *   Parses uploaded CVs (text extraction needed in API).
    *   Uses Gemini to extract contact info, work history, education, skills, projects, etc.
    *   Standardizes and structures the extracted CV data.
3.  **Orchestration Agent (for Scoring):**
    *   Receives JD and CV text.
    *   Orchestrates a two-step process:
        *   Step 1: Uses Gemini to extract skills from both JD and CV.
        *   Step 2: Uses Gemini to synthesize a score (0-100) and explanation based on full texts and extracted skills.
    *   Provides explainability for the score.

**(Potential Later) RAG Agent:** A specialized agent or integrated component focused on retrieving relevant information from documents or external sources to ground analysis and scoring.

**High-Level Technology Stack:**

*   **Backend:** Python (FastAPI) - Modern, high-performance framework suitable for AI/ML integration. **[Selected & Setup]**
*   **Frontend:** React, Vue, or Angular (To be decided later) - Modern JavaScript frameworks for building interactive UIs.
*   **Database:** PostgreSQL (Implicitly chosen via SQLAlchemy setup) - Based on data structure needs. **[Setup with SQLAlchemy & Alembic]**
*   **AI:** Google Gemini API (via Python client library using `google-generativeai`). **[Integrated & Tested]**
*   **Deployment:** Docker, Cloud Platform (GCP, AWS, Azure).

**Roadmap Status:**

**Phase 1: Foundation & Setup (Est. 1-2 Weeks) - [COMPLETED]**

1.    ✅ **Project Initialization:** Project structure, Git repo, `README.md`, `ROADMAP.md`.
2.    ✅ **Environment:** `python-dotenv`, `requirements.txt`.
3.    ✅ **Backend Choice & Setup:** FastAPI setup.
4.    ✅ **Basic API:** Health check endpoint created.
5.    ✅ **Gemini Connection Test:** Script created and successfully tested with real API key and model (`gemini-2.0-flash-lite`).

**Phase 2: Core Data Handling & JD Agent (Est. 3-4 Weeks) - [Mostly Complete]**

1.    ✅ **Database Setup:** SQLAlchemy models (`Job`, `Candidate`, `Score`), Alembic migrations setup.
2.    ✅ **File Upload API:** API endpoint for uploading JD files (`/api/v1/jobs/upload`).
3.    ✅ **JD Parsing:** Basic text extraction utility (`extract_text_from_file`) added.
4.    ✅ **JD Analysis Agent (v1):**
    *   ✅ Integrated Gemini API calls with function calling.
    *   ✅ Developed prompts for extracting structured info.
    *   ✅ Stores structured JD analysis results in the database (`Job.analysis` field).
    *   ✅ Tested successfully.
5.  ✅ **API Expansion:** Endpoint to trigger JD analysis (`/api/v1/jobs/{job_id}/analyze`).

**Phase 3: CV Agent & Basic Matching (Est. 3-4 Weeks) - [Partially Complete]**

1.    🚧 **CV Upload API:** API endpoints for uploading CV files (Need implementation).
2.    🚧 **CV Parsing:** Leverage existing `extract_text_from_file` in the CV upload API.
3.    ✅ **CV Analysis Agent (v1):**
    *   ✅ Integrated Gemini API calls.
    *   ✅ Developed prompts for extracting structured info.
    *   ✅ Stores structured CV analysis results (`Candidate.analysis` field).
    *   ✅ Tested successfully.
4.  ✅ **Scoring & Matching Agent (OrchestrationAgent v1):**
    *   ✅ Develops two-step logic (skill extraction, score synthesis).
    *   ✅ Implements scoring mechanism using Gemini.
    *   ✅ Defines `Score` schema/model.
    *   ✅ Tested successfully.
5.  ✅ **API Expansion:** Endpoints to trigger scoring (`/api/v1/scores/`) and retrieve scores.

**Phase 4: Frontend Development & Integration (Est. 4-6 Weeks) - [Not Started]**

1.    ⚪ **Frontend Setup:** Choose and set up the frontend framework.
2.    ⚪ **UI Components:** Build UI for:
    *   Uploading JDs and CVs.
    *   Viewing lists of Jobs and Candidates.
    *   Displaying analysis results and scores for a specific match (including professional report styling).
3.  ⚪ **API Integration:** Connect frontend components to the backend APIs.

**Phase 5: SOTA Refinement & Advanced Features (Ongoing)**

1.  **Agent Improvement:** Refine prompts, potentially fine-tune models, improve parsing robustness.
2.  **RAG Implementation:** Integrate SOTA RAG techniques for improved grounding and explainability.
3.  **Advanced Scoring:** Research and implement more sophisticated scoring techniques (semantic search, vector embeddings, graph-based matching). Enhance explainability.
4.  **Testing:** Implement comprehensive unit, integration, and potentially end-to-end tests.
5.  **Deployment Strategy:** Set up CI/CD pipelines for different environments.
6.  **Recruiter-Focused UX Enhancements:**
    *   **Advanced Candidate Ranking/Filtering:** Rank/filter based on specific criteria.
    *   **Improved Batch Processing:** Efficiently process multiple CVs for one job.
    *   **Side-by-Side Comparison UI:** Compare top candidates against JD.
    *   **Interview Question Generation:** Suggest questions based on analysis.
    *   **Market Insights Expansion:** Enhance JD analysis with real-time salary/skill data.
    *   **(Optional/Careful) "Red Flag" Identification:** Highlight potential concerns.
    *   **Internal Notes/Collaboration:** Allow recruiter annotations.
    *   **CV Highlighting:** Visually link CV sections to JD requirements.
    *   **Professional Report Generation (UI/PDF):** Standardized, downloadable reports.
7.  **Future Considerations:**
    *   **ATS Integration:** Plan for connecting with Applicant Tracking Systems. 