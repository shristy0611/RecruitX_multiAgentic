import pytest
import os
import sys
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from recruitx_app.services.scoring_service import ScoringService
from recruitx_app.services.job_service import JobService
from recruitx_app.services.candidate_service import CandidateService
from recruitx_app.schemas.job import JobCreate, JobAnalysis
from recruitx_app.schemas.candidate import CandidateCreate, CandidateAnalysis, WorkExperience, Education
from recruitx_app.schemas.score import Score

# Test data
TEST_JD = """
Job Title: Senior Python Developer

About Us:
We are a leading fintech company developing innovative solutions for the financial industry.

Responsibilities:
- Design, build and maintain efficient, reusable, and reliable Python code
- Implement data storage solutions using PostgreSQL, MongoDB, and Redis
- Develop and integrate RESTful APIs
- Collaborate with frontend developers to integrate user-facing elements

Requirements:
- 5+ years of experience in Python development
- Proficient with Python web frameworks (Django or Flask)
- Strong knowledge of ORM libraries and SQL/NoSQL databases
- Experience with Docker, Kubernetes, and CI/CD pipelines
"""

TEST_CV = """
JOHN DEVELOPER
Software Engineer
john.developer@email.com | (555) 123-4567 | San Francisco, CA

SUMMARY
Experienced software engineer with 6+ years of expertise in Python development, specializing in web applications and cloud deployment. Strong background in developing scalable microservices and RESTful APIs.

SKILLS
Programming Languages: Python, JavaScript, SQL
Frameworks: Django, Flask, React
Databases: PostgreSQL, MongoDB, Redis
DevOps: Docker, Kubernetes, Jenkins, AWS

WORK EXPERIENCE
Senior Python Developer | FinanceTech Inc. | Jan 2021 - Present
- Developed microservices architecture using Django and Flask
- Implemented database solutions with PostgreSQL and MongoDB
- Created RESTful APIs for frontend integration
- Containerized applications using Docker and Kubernetes
"""

# Mock responses
MOCK_JOB_ANALYSIS = JobAnalysis(
    required_skills=["Python", "Django", "Flask", "PostgreSQL", "MongoDB", "REST API"],
    preferred_skills=["Docker", "Kubernetes", "CI/CD"],
    minimum_experience="5+ years",
    education="Bachelor's Degree in CS",
    responsibilities=["Develop Python code", "Implement databases", "Create APIs"],
    job_type="Full-time",
    seniority_level="Senior",
    market_insights={
        "skill_demand": {
            "high_demand_skills": ["Python", "Django", "Kubernetes"],
            "trending_skills": ["FastAPI", "GraphQL"]
        },
        "salary_insights": "$120K - $150K per year",
        "industry_outlook": "Strong growth in fintech sector"
    },
    reasoning="Analysis based on JD content and industry standards"
)

MOCK_CV_ANALYSIS = CandidateAnalysis(
    candidate_id=1,
    contact_info={
        "name": "John Developer",
        "email": "john.developer@email.com",
        "phone": "(555) 123-4567",
        "location": "San Francisco, CA"
    },
    summary="Experienced software engineer with 6+ years of expertise in Python development",
    skills=["Python", "JavaScript", "SQL", "Django", "Flask", "React", "PostgreSQL", "MongoDB", "Redis", "Docker", "Kubernetes", "AWS"],
    work_experience=[
        WorkExperience(
            company="FinanceTech Inc.",
            title="Senior Python Developer",
            dates="Jan 2021 - Present",
            responsibilities=[
                "Developed microservices architecture using Django and Flask",
                "Implemented database solutions with PostgreSQL and MongoDB",
                "Created RESTful APIs for frontend integration"
            ]
        )
    ],
    education=[
        Education(
            institution="Stanford University",
            degree="Bachelor of Science",
            field_of_study="Computer Science",
            graduation_date="2018"
        )
    ],
    certifications=["AWS Certified Developer"],
    projects=[],
    languages=["English"],
    overall_profile="Strong Python developer with relevant experience"
)

MOCK_SCORE_DETAILS = {
    "skills_match": {
        "score": 92,
        "matching_skills": ["Python", "Django", "Flask", "PostgreSQL", "MongoDB"],
        "missing_skills": ["REST API"],
        "additional_skills": ["JavaScript", "SQL", "React", "Redis", "AWS"]
    },
    "experience_match": {
        "score": 95,
        "evaluation": "The candidate has 6+ years of Python experience, exceeding the minimum requirement of 5+ years"
    },
    "education_match": {
        "score": 100,
        "evaluation": "The candidate has a Bachelor's degree in Computer Science, meeting the education requirement"
    },
    "responsibility_match": {
        "score": 90,
        "matching_responsibilities": [
            {"job": "Develop Python code", "candidate": "Developed microservices architecture using Django and Flask"},
            {"job": "Implement databases", "candidate": "Implemented database solutions with PostgreSQL and MongoDB"},
            {"job": "Create APIs", "candidate": "Created RESTful APIs for frontend integration"}
        ]
    }
}

@pytest.mark.asyncio
class TestScoringPipeline:
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        mock_db = MagicMock()
        return mock_db
    
    @pytest.fixture
    def job_service(self):
        """Create a JobService instance for testing."""
        return JobService()
    
    @pytest.fixture
    def candidate_service(self):
        """Create a CandidateService instance for testing."""
        return CandidateService()
    
    @pytest.fixture
    def scoring_service(self):
        """Create a ScoringService instance for testing."""
        return ScoringService()
    
    @patch('recruitx_app.agents.jd_analysis_agent.JDAnalysisAgent.analyze_job_description')
    @patch('recruitx_app.agents.cv_analysis_agent.CVAnalysisAgent.analyze_cv')
    @patch('recruitx_app.services.scoring_service.ScoringService._get_scoring_agent')
    async def test_full_scoring_pipeline(self, mock_scoring_agent, mock_analyze_cv, mock_analyze_jd, 
                                        mock_db, job_service, candidate_service, scoring_service):
        """Test the full scoring pipeline integration with all components."""
        
        # Setup mock responses for agent methods
        mock_analyze_jd.return_value = MOCK_JOB_ANALYSIS
        mock_analyze_cv.return_value = MOCK_CV_ANALYSIS
        
        # Setup mock for scoring agent
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.overall_score = 93
        mock_response.explanation = "The candidate is an excellent match for the position"
        mock_response.details = MOCK_SCORE_DETAILS
        
        async def mock_generate(*args, **kwargs):
            return mock_response
            
        mock_agent.generate_score.side_effect = mock_generate
        mock_scoring_agent.return_value = mock_agent
        
        # Create test job data
        job_create = JobCreate(
            title="Senior Python Developer",
            company="Test Company",
            location="San Francisco",
            description_raw=TEST_JD
        )
        mock_job = MagicMock()
        mock_job.id = 1
        mock_job.description_raw = TEST_JD
        
        # Mock job creation
        with patch.object(job_service, 'create_job', return_value=mock_job):
            job = job_service.create_job(mock_db, job_data=job_create)
        
        # Create test candidate data
        candidate_create = CandidateCreate(
            name="John Developer",
            email="john.developer@email.com",
            phone="(555) 123-4567",
            resume_raw=TEST_CV
        )
        mock_candidate = MagicMock()
        mock_candidate.id = 1
        mock_candidate.resume_raw = TEST_CV
        
        # Mock candidate creation
        with patch.object(candidate_service, 'create_candidate', return_value=mock_candidate):
            candidate = candidate_service.create_candidate(mock_db, candidate_data=candidate_create)
        
        # Test the full pipeline
        
        # 1. Analyze job
        job_analysis = await job_service.analyze_job(mock_db, job_id=job.id)
        assert job_analysis == MOCK_JOB_ANALYSIS
        mock_analyze_jd.assert_called_once_with(job.id, job.description_raw)
        
        # 2. Analyze CV
        cv_analysis = await candidate_service.analyze_cv(mock_db, candidate_id=candidate.id)
        assert cv_analysis == MOCK_CV_ANALYSIS
        mock_analyze_cv.assert_called_once_with(candidate.resume_raw, candidate.id)
        
        # 3. Generate score
        score = await scoring_service.generate_score(mock_db, job_id=job.id, candidate_id=candidate.id)
        
        # Verify score results
        assert score.overall_score == 93
        assert score.explanation == "The candidate is an excellent match for the position"
        assert score.details == MOCK_SCORE_DETAILS
        
        # Verify the integration flow
        mock_scoring_agent.assert_called_once()
        mock_agent.generate_score.assert_called_once()
        
        # Verify correct arguments were passed to generate_score
        call_args = mock_agent.generate_score.call_args[1]
        assert call_args["job_analysis"] == MOCK_JOB_ANALYSIS
        assert call_args["cv_analysis"] == MOCK_CV_ANALYSIS 