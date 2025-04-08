import pytest
import os
import sys
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from recruitx_app.services.candidate_service import CandidateService
from recruitx_app.models.candidate import Candidate
from recruitx_app.schemas.candidate import CandidateAnalysis, ContactInfo, WorkExperience, Education

class TestCandidateService:
    """Test class for CandidateService."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session for testing."""
        mock_session = MagicMock()
        return mock_session
    
    @pytest.fixture
    def candidate_service(self):
        """Create a CandidateService instance for testing."""
        # Create a service with a mock CV agent
        with patch('recruitx_app.services.candidate_service.CVAnalysisAgent') as mock_cv_agent_class:
            service = CandidateService()
            # Keep a reference to the mocked agent for assertions
            service._mock_cv_agent = mock_cv_agent_class.return_value
            return service
    
    @pytest.fixture
    def sample_candidate(self):
        """Create a sample candidate for testing."""
        return Candidate(
            id=1,
            name="John Doe",
            email="john.doe@example.com",
            phone="+1234567890",
            resume_raw="Sample resume content with relevant experience and skills."
        )
    
    @pytest.fixture
    def sample_candidate_analysis(self):
        """Create a sample candidate analysis result for testing."""
        return CandidateAnalysis(
            candidate_id=1,
            contact_info={
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+1234567890",
            },
            summary="Experienced software engineer with 5+ years of experience.",
            skills=["Python", "JavaScript", "SQL", "AWS"],
            work_experience=[
                {
                    "company": "Tech Company",
                    "title": "Senior Developer",
                    "dates": "2020-Present",
                    "responsibilities": ["Led development team", "Implemented CI/CD"]
                }
            ],
            education=[
                {
                    "institution": "University of Technology",
                    "degree": "Bachelor of Science",
                    "field_of_study": "Computer Science",
                    "graduation_date": "2015"
                }
            ],
            certifications=["AWS Certified Developer"],
            projects=[],
            languages=["English"],
            overall_profile="Strong technical background with good leadership experience."
        )
    
    def test_get_candidates(self, candidate_service, mock_db_session, sample_candidate):
        """Test get_candidates method."""
        # Configure mock to return a list of candidates
        mock_candidates = [sample_candidate, MagicMock()]
        mock_db_session.query.return_value.offset.return_value.limit.return_value.all.return_value = mock_candidates
        
        # Call the method
        result = candidate_service.get_candidates(mock_db_session)
        
        # Verify results
        assert result == mock_candidates
        mock_db_session.query.assert_called_once_with(Candidate)
    
    def test_get_candidates_with_pagination(self, candidate_service, mock_db_session):
        """Test get_candidates method with pagination parameters."""
        # Configure mock
        mock_db_session.query.return_value.offset.return_value.limit.return_value.all.return_value = []
        
        # Call the method with pagination
        candidate_service.get_candidates(mock_db_session, skip=10, limit=20)
        
        # Verify correct pagination parameters were used
        mock_db_session.query.assert_called_once_with(Candidate)
        # Use assert_called_with instead of assert_called_once_with for chained methods
        mock_db_session.query.return_value.offset.assert_called_with(10)
        mock_db_session.query.return_value.offset.return_value.limit.assert_called_with(20)
    
    def test_get_candidate(self, candidate_service, mock_db_session, sample_candidate):
        """Test get_candidate method."""
        # Configure mock to return a specific candidate
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_candidate
        
        # Call the method
        result = candidate_service.get_candidate(mock_db_session, candidate_id=1)
        
        # Verify results
        assert result == sample_candidate
        mock_db_session.query.assert_called_once_with(Candidate)
    
    def test_get_candidate_not_found(self, candidate_service, mock_db_session):
        """Test get_candidate method when candidate is not found."""
        # Configure mock to return None
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        # Call the method
        result = candidate_service.get_candidate(mock_db_session, candidate_id=999)
        
        # Verify results
        assert result is None
        # Only verify that query was called with Candidate
        mock_db_session.query.assert_called_once_with(Candidate)
    
    def test_create_candidate(self, candidate_service, mock_db_session):
        """Test create_candidate method."""
        # Sample candidate data
        candidate_data = {
            "name": "Jane Smith",
            "email": "jane.smith@example.com",
            "phone": "+1987654321",
            "resume_raw": "Sample resume content."
        }
        
        # Call the method
        result = candidate_service.create_candidate(mock_db_session, candidate_data)
        
        # Verify the DB operations were called correctly
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
        
        # Verify the candidate object was created with the right data
        added_candidate = mock_db_session.add.call_args[0][0]
        assert isinstance(added_candidate, Candidate)
        assert added_candidate.name == candidate_data["name"]
        assert added_candidate.email == candidate_data["email"]
        assert added_candidate.phone == candidate_data["phone"]
        assert added_candidate.resume_raw == candidate_data["resume_raw"]
    
    @pytest.mark.asyncio
    async def test_analyze_cv_successful(self, candidate_service, mock_db_session, sample_candidate, sample_candidate_analysis):
        """Test analyze_cv method with successful analysis."""
        # Configure mocks
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_candidate
        
        # Mock the CV agent's analyze_cv method
        candidate_service.cv_agent.analyze_cv = AsyncMock(return_value=sample_candidate_analysis)
        
        # Mock the vector_db_service
        with patch('recruitx_app.services.candidate_service.vector_db_service.add_document_chunks', 
                   new_callable=AsyncMock) as mock_add_chunks, \
             patch('recruitx_app.services.candidate_service.split_text') as mock_split_text:
            
            # Configure mock split_text to return some chunks
            mock_split_text.return_value = ["Chunk 1", "Chunk 2"]
            
            # Configure mock add_document_chunks to indicate success
            mock_add_chunks.return_value = True
            
            # Call the method
            result = await candidate_service.analyze_cv(mock_db_session, candidate_id=1)
            
            # Verify CV agent was called with the right parameters
            candidate_service.cv_agent.analyze_cv.assert_called_once_with(
                cv_text=sample_candidate.resume_raw,
                candidate_id=1
            )
            
            # Verify DB updates
            assert mock_db_session.commit.called
            assert mock_db_session.refresh.called
            
            # Verify result is the expected analysis
            assert result == sample_candidate_analysis
            
            # Verify chunking and indexing
            mock_split_text.assert_called_once_with(sample_candidate.resume_raw)
            mock_add_chunks.assert_called_once()
            # Verify parameters to add_document_chunks
            call_args = mock_add_chunks.call_args[1]
            assert "documents" in call_args
            assert "metadatas" in call_args
            assert "ids" in call_args
            assert len(call_args["documents"]) == 2
            assert len(call_args["metadatas"]) == 2
            assert len(call_args["ids"]) == 2
    
    @pytest.mark.asyncio
    async def test_analyze_cv_candidate_not_found(self, candidate_service, mock_db_session):
        """Test analyze_cv method when candidate is not found."""
        # Configure mock to return None
        mock_db_session.query().filter().first.return_value = None
        
        # Call the method
        result = await candidate_service.analyze_cv(mock_db_session, candidate_id=999)
        
        # Verify CV agent was not called
        assert not hasattr(candidate_service.cv_agent, 'analyze_cv') or not candidate_service.cv_agent.analyze_cv.called
        
        # Verify result is None
        assert result is None
    
    @pytest.mark.asyncio
    async def test_analyze_cv_no_resume(self, candidate_service, mock_db_session):
        """Test analyze_cv method when candidate has no resume."""
        # Configure mock to return a candidate with no resume
        candidate_without_resume = MagicMock(spec=Candidate)
        candidate_without_resume.id = 1
        candidate_without_resume.resume_raw = None
        mock_db_session.query().filter().first.return_value = candidate_without_resume
        
        # Call the method
        result = await candidate_service.analyze_cv(mock_db_session, candidate_id=1)
        
        # Verify CV agent was not called
        assert not hasattr(candidate_service.cv_agent, 'analyze_cv') or not candidate_service.cv_agent.analyze_cv.called
        
        # Verify result is None
        assert result is None
    
    @pytest.mark.asyncio
    async def test_analyze_cv_analysis_failed(self, candidate_service, mock_db_session, sample_candidate):
        """Test analyze_cv method when analysis fails."""
        # Configure mocks
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_candidate
        
        # Mock the CV agent to return None (analysis failed)
        candidate_service.cv_agent.analyze_cv = AsyncMock(return_value=None)
        
        # Call the method
        result = await candidate_service.analyze_cv(mock_db_session, candidate_id=1)
        
        # Verify CV agent was called
        candidate_service.cv_agent.analyze_cv.assert_called_once()
        
        # Verify DB was not updated
        assert not mock_db_session.commit.called
        
        # Verify result is None
        assert result is None
    
    @pytest.mark.asyncio
    async def test_analyze_cv_db_update_error(self, candidate_service, mock_db_session, sample_candidate, sample_candidate_analysis):
        """Test analyze_cv method when database update fails."""
        # Configure mocks
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_candidate
        
        # Mock the CV agent
        candidate_service.cv_agent.analyze_cv = AsyncMock(return_value=sample_candidate_analysis)
        
        # Configure the DB session to raise an exception during commit
        mock_db_session.commit.side_effect = Exception("Database error")
        
        # Call the method
        result = await candidate_service.analyze_cv(mock_db_session, candidate_id=1)
        
        # Verify rollback was called
        mock_db_session.rollback.assert_called_once()
        
        # Verify result is None due to the error
        assert result is None
    
    @pytest.mark.asyncio
    async def test_analyze_cv_indexing_error(self, candidate_service, mock_db_session, sample_candidate, sample_candidate_analysis):
        """Test analyze_cv method when vector indexing fails."""
        # Configure mocks
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_candidate
        
        # Mock the CV agent
        candidate_service.cv_agent.analyze_cv = AsyncMock(return_value=sample_candidate_analysis)
        
        # Mock the vector_db_service to fail
        with patch('recruitx_app.services.candidate_service.vector_db_service.add_document_chunks', 
                   new_callable=AsyncMock) as mock_add_chunks, \
             patch('recruitx_app.services.candidate_service.split_text') as mock_split_text:
            
            # Configure mock split_text to return some chunks
            mock_split_text.return_value = ["Chunk 1", "Chunk 2"]
            
            # Configure mock add_document_chunks to indicate failure
            mock_add_chunks.return_value = False
            
            # Call the method
            result = await candidate_service.analyze_cv(mock_db_session, candidate_id=1)
            
            # Verify result is still the analysis despite indexing failure
            assert result == sample_candidate_analysis
    
    @pytest.mark.asyncio
    async def test_analyze_cv_chunking_error(self, candidate_service, mock_db_session, sample_candidate, sample_candidate_analysis):
        """Test analyze_cv method when text chunking fails or returns no chunks."""
        # Configure mocks
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_candidate
        
        # Mock the CV agent
        candidate_service.cv_agent.analyze_cv = AsyncMock(return_value=sample_candidate_analysis)
        
        # Mock split_text to return empty list
        with patch('recruitx_app.services.candidate_service.split_text') as mock_split_text:
            mock_split_text.return_value = []
            
            # Call the method
            result = await candidate_service.analyze_cv(mock_db_session, candidate_id=1)
            
            # Verify result is still the analysis despite chunking failure
            assert result == sample_candidate_analysis 