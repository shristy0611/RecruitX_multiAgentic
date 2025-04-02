import pytest
import os
import sys
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional
from pydantic import ValidationError

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from recruitx_app.services.scoring_service import ScoringService
from recruitx_app.schemas.job import JobAnalysis, MarketInsights, SkillDemand
from recruitx_app.schemas.candidate import CandidateAnalysis
from recruitx_app.models.job import Job
from recruitx_app.models.candidate import Candidate

# Define Score class for testing
class Score:
    def __init__(
        self, 
        job_id: int, 
        candidate_id: int, 
        overall_score: float, 
        explanation: str,
        details: Dict[str, Any] = None
    ):
        self.job_id = job_id
        self.candidate_id = candidate_id
        self.overall_score = overall_score
        self.explanation = explanation
        self.details = details or {}

# Mock data for testing
MOCK_JOB_ANALYSIS = JobAnalysis(
    job_id=1,
    required_skills=["Python", "SQL", "Machine Learning"],
    preferred_skills=["Docker", "AWS"],
    minimum_experience="3+ years",
    job_description="This is a test job description",
    market_insights=MarketInsights(
        skill_demand=SkillDemand(
            high_demand_skills=["Python", "Machine Learning"],
            trending_skills=["Rust", "Kubernetes"]
        ),
        salary_insights="$90K - $120K",
        industry_outlook="Strong growth expected in the AI/ML sector"
    )
)

MOCK_CV_ANALYSIS = CandidateAnalysis(
    candidate_id=1,
    contact_info={
        "name": "John Developer",
        "email": "john.developer@example.com",
        "phone": "(555) 123-4567",
        "location": "New York, NY"
    },
    summary="Software engineer with 5 years of experience",
    skills=["Python", "JavaScript", "SQL", "Docker"],
    work_experience=[
        {
            "company": "Example Corp",
            "role": "Software Engineer",
            "dates": "2019-2023",
            "skills": ["Python", "SQL", "Docker"],
            "description": "Developed various software applications."
        },
        {
            "company": "Tech Startup",
            "role": "Junior Developer",
            "dates": "2017-2019",
            "skills": ["JavaScript", "HTML", "CSS"],
            "description": "Helped build websites and web applications."
        }
    ],
    education=[
        {
            "degree": "Bachelor of Science",
            "field": "Computer Science",
            "institution": "Example University",
            "dates": "2015-2019"
        }
    ],
    certifications=["AWS Certified Developer"],
    projects=[],
    languages=["English"],
    overall_profile="Strong developer with relevant experience",
    cv_text="This is a test CV text with relevant skills and experience."
)

class TestScoringService:
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.refresh = MagicMock()
        mock_session.query = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None
        return mock_session
    
    @pytest.fixture
    def scoring_service(self, mock_db_session):
        """Create a ScoringService with mock dependencies."""
        service = ScoringService()
        
        # Mock the orchestration agent
        service.orchestration_agent = AsyncMock()
        service.orchestration_agent.synthesize_score = AsyncMock(return_value={
            "overall_score": 85.5,
            "explanation": "The candidate is a good match for this job.",
            "skills_match": {
                "score": 90,
                "matches": ["Python", "SQL"],
                "missing": ["Machine Learning"]
            },
            "experience_match": {
                "score": 80,
                "evaluation": "Meets requirements"
            },
            "education_match": {
                "score": 85,
                "evaluation": "Relevant degree"
            }
        })
        
        # Mock the JD analysis agent
        service.jd_analysis_agent = AsyncMock()
        service.jd_analysis_agent.decompose_job_description = AsyncMock(return_value=[MagicMock(
            is_required=True,
            model_dump=lambda: {"facet_id": 1, "name": "Python", "is_required": True}
        )])
        
        return service
    
    @pytest.mark.asyncio
    async def test_generate_score_successful(self, scoring_service, mock_db_session):
        """Test successful score generation."""
        # Mock the database query to return a job and candidate
        mock_job = MagicMock(spec=Job)
        mock_job.id = 1
        mock_job.title = "Software Engineer"
        mock_job.description_raw = "Test job description"
        
        mock_candidate = MagicMock(spec=Candidate)
        mock_candidate.id = 1
        mock_candidate.name = "Test Candidate"
        mock_candidate.resume_raw = "Test CV content"
        
        # Set up database mock to return our mock objects
        mock_db_session.query.return_value.filter.return_value.first.side_effect = [
            mock_job,
            mock_candidate
        ]
        
        # Mock the agentic_rag_service
        with patch('recruitx_app.services.scoring_service.agentic_rag_service') as mock_rag:
            # Configure mock responses
            mock_rag.iterative_retrieve_and_validate = AsyncMock(return_value={0: "Mock evidence"})
            mock_rag.enrich_evidence_with_external_data = AsyncMock(return_value={
                "external_data": {"salary_benchmark": {"data": "sample"}},
                "facet_external_data": {"1": {"data": "sample"}}
            })
            
            # Call the method
            result = await scoring_service.generate_score(
                db=mock_db_session,
                job_id=1,
                candidate_id=1
            )
            
            # Verify database queries and agent calls
            assert mock_db_session.query.call_count == 2
            
            # Verify JD agent was called
            scoring_service.jd_analysis_agent.decompose_job_description.assert_called_once_with(
                job_id=1,
                job_description="Test job description"
            )
            
            # Verify agentic_rag_service calls
            mock_rag.iterative_retrieve_and_validate.assert_called_once()
            mock_rag.enrich_evidence_with_external_data.assert_called_once()
            
            # Verify orchestration agent was called with correct arguments
            scoring_service.orchestration_agent.synthesize_score.assert_called_once()
            call_args = scoring_service.orchestration_agent.synthesize_score.call_args[1]
            assert call_args["job_description"] == "Test job description"
            assert call_args["candidate_resume"] == "Test CV content"
            assert call_args["candidate_id"] == 1
            
            # Verify database operations
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()
            mock_db_session.refresh.assert_called_once()
            
            # Verify the result
            assert result is not None
            assert result.job_id == 1
            assert result.candidate_id == 1
            assert result.overall_score == 85.5
            assert "The candidate is a good match for this job." in result.explanation
    
    @pytest.mark.asyncio
    async def test_error_during_job_analysis(self, scoring_service, mock_db_session):
        """Test handling of error during job analysis."""
        # Mock the database query to return a job and candidate
        mock_job = MagicMock(spec=Job)
        mock_job.id = 1
        mock_job.description_raw = "Test job description"
        
        mock_candidate = MagicMock(spec=Candidate)
        mock_candidate.id = 1
        mock_candidate.resume_raw = "Test CV content"
        
        # Set up database mock to return our mock objects
        mock_db_session.query.return_value.filter.return_value.first.side_effect = [
            mock_job,
            mock_candidate
        ]
        
        # Set up error in JD analysis
        scoring_service.jd_analysis_agent.decompose_job_description.return_value = None
        
        # Call the method
        result = await scoring_service.generate_score(
            db=mock_db_session,
            job_id=1,
            candidate_id=1
        )
        
        # Verify JD agent was called
        scoring_service.jd_analysis_agent.decompose_job_description.assert_called_once()
        
        # Verify error handling - a score should still be created with error information
        assert result is not None
        assert result.overall_score == 0.0
        assert "Failed during job description decomposition" in result.explanation
        
        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_exception_handling(self, scoring_service, mock_db_session):
        """Test handling of exceptions during score generation."""
        # Mock the database query to return a job and candidate
        mock_job = MagicMock(spec=Job)
        mock_job.id = 1
        mock_job.title = "Software Engineer"
        mock_job.description_raw = "Test job description"
        
        mock_candidate = MagicMock(spec=Candidate)
        mock_candidate.id = 1
        mock_candidate.name = "Test Candidate"
        mock_candidate.resume_raw = "Test CV content"
        
        # Set up database mock to return our mock objects
        mock_db_session.query.return_value.filter.return_value.first.side_effect = [
            mock_job,
            mock_candidate
        ]
        
        # Set up error in JD analysis
        scoring_service.jd_analysis_agent.decompose_job_description.side_effect = Exception("Test exception")
        
        # Call the method
        result = await scoring_service.generate_score(
            db=mock_db_session,
            job_id=1,
            candidate_id=1
        )
        
        # Verify exception handling
        assert result is None
        mock_db_session.rollback.assert_called_once()
        
        # Verify no score was persisted
        mock_db_session.add.assert_not_called()
        mock_db_session.commit.assert_not_called() 