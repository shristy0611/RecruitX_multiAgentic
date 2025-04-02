import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Generator, Dict, Any
from unittest.mock import MagicMock

# Adjust the Python path to include the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from recruitx_app.main import app
from recruitx_app.core.config import settings
from recruitx_app.core.database import Base, get_db
from recruitx_app.agents.jd_analysis_agent import JDAnalysisAgent
from recruitx_app.agents.cv_analysis_agent import CVAnalysisAgent

# Use in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Sample data for tests
SAMPLE_JD = """
Job Title: Senior Python Developer

About Us:
We are a leading fintech company developing innovative solutions for the financial industry.

Responsibilities:
- Design, build and maintain efficient, reusable, and reliable Python code
- Implement data storage solutions using PostgreSQL, MongoDB, and Redis
- Develop and integrate RESTful APIs

Requirements:
- 5+ years of experience in Python development
- Proficient with Python web frameworks (Django or Flask)
- Strong knowledge of ORM libraries and SQL/NoSQL databases
"""

SAMPLE_CV = """
JOHN DEVELOPER
Software Engineer
john.developer@email.com | (555) 123-4567 | San Francisco, CA

SUMMARY
Experienced software engineer with 6+ years of expertise in Python development.

SKILLS
Programming Languages: Python, JavaScript, SQL
Frameworks: Django, Flask, React
Databases: PostgreSQL, MongoDB

WORK EXPERIENCE
Senior Python Developer | FinanceTech Inc. | Jan 2021 - Present
- Developed microservices architecture using Django and Flask
- Implemented database solutions with PostgreSQL and MongoDB
"""


# Set up the database once for all tests
@pytest.fixture(scope="session")
def db_engine():
    # Create the database tables
    Base.metadata.create_all(bind=engine)
    yield engine
    # Drop all tables after tests are done
    Base.metadata.drop_all(bind=engine)


# Create a fresh database session for each test
@pytest.fixture
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


# Override the get_db dependency
@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# Mock JD Analysis Agent
@pytest.fixture
def mock_jd_agent():
    mock_agent = MagicMock(spec=JDAnalysisAgent)
    
    # Configure mock response for get_relevant_context
    async def mock_get_relevant_context(*args, **kwargs):
        return "Mock JD context from similar job descriptions"
    mock_agent.get_relevant_context.side_effect = mock_get_relevant_context
    
    # Configure mock response for analyze_job_description
    async def mock_analyze(*args, **kwargs):
        from recruitx_app.schemas.job import JobAnalysis
        return JobAnalysis(
            required_skills=["Python", "Django", "PostgreSQL"],
            preferred_skills=["React", "Docker"],
            minimum_experience="5+ years",
            education="Bachelor's in Computer Science",
            responsibilities=["Develop backend services", "Database management"],
            job_type="Full-time",
            seniority_level="Senior",
            market_insights={
                "skill_demand": {
                    "high_demand_skills": ["Python", "Django"],
                    "trending_skills": ["FastAPI", "Kubernetes"] 
                },
                "salary_insights": "$120K - $150K",
                "industry_outlook": "Strong growth in fintech sector"
            },
            reasoning="Analysis based on job requirements and industry standards"
        )
    mock_agent.analyze_job_description.side_effect = mock_analyze
    
    return mock_agent


# Mock CV Analysis Agent
@pytest.fixture
def mock_cv_agent():
    mock_agent = MagicMock(spec=CVAnalysisAgent)
    
    # Configure mock response for get_relevant_context
    async def mock_get_relevant_context(*args, **kwargs):
        return "Mock CV context from similar profiles and job descriptions"
    mock_agent.get_relevant_context.side_effect = mock_get_relevant_context
    
    # Configure mock response for analyze_cv
    async def mock_analyze(*args, **kwargs):
        from recruitx_app.schemas.candidate import CandidateAnalysis, WorkExperience, Education
        return CandidateAnalysis(
            candidate_id=1,
            contact_info={
                "name": "John Developer",
                "email": "john.developer@email.com",
                "phone": "(555) 123-4567",
                "location": "San Francisco, CA"
            },
            summary="Experienced software engineer with 6+ years of expertise in Python development",
            skills=["Python", "Django", "Flask", "PostgreSQL", "MongoDB"],
            work_experience=[
                WorkExperience(
                    company="FinanceTech Inc.",
                    title="Senior Python Developer",
                    dates="Jan 2021 - Present",
                    responsibilities=["Developed microservices", "Implemented databases"]
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
    mock_agent.analyze_cv.side_effect = mock_analyze
    
    return mock_agent


# Sample data fixtures
@pytest.fixture
def sample_jd():
    return SAMPLE_JD


@pytest.fixture
def sample_cv():
    return SAMPLE_CV 