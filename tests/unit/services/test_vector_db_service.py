import pytest
import os
import sys
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from recruitx_app.services.vector_db_service import VectorDBService

# Test data
TEST_DOCUMENTS = [
    "This is a test job description for a Python developer position.",
    "This is a test CV for a candidate with Python experience."
]

TEST_METADATA = [
    {"doc_type": "job", "job_id": 1, "timestamp": "2025-04-01"},
    {"doc_type": "candidate", "candidate_id": 1, "timestamp": "2025-04-01"}
]

MOCK_QUERY_RESULTS = {
    'ids': [['doc1', 'doc2']],
    'documents': [TEST_DOCUMENTS],
    'metadatas': [TEST_METADATA],
    'distances': [[0.1, 0.2]]
}

class TestVectorDBService:
    
    @pytest.fixture
    def mock_chromadb(self):
        """Create a mock ChromaDB client."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        # Set up the client to return our mock collection
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Configure the collection's add method
        mock_collection.add.return_value = None
        
        # Configure the collection's query method
        mock_collection.query.return_value = MOCK_QUERY_RESULTS
        
        return mock_client
    
    @pytest.fixture
    def vector_db_service(self, mock_chromadb):
        """Create a VectorDBService with a mock ChromaDB client."""
        with patch('chromadb.PersistentClient', return_value=mock_chromadb):
            service = VectorDBService()
            service._client = mock_chromadb
            service._collection = mock_chromadb.get_or_create_collection()
            return service
    
    def test_initialize_client(self, vector_db_service, mock_chromadb):
        """Test the client initialization process."""
        # Reset the client
        vector_db_service._client = None
        
        # Call the initialize method
        vector_db_service._initialize_client()
        
        # Verify that the client was created
        assert vector_db_service._client is not None
    
    def test_get_collection(self, vector_db_service, mock_chromadb):
        """Test getting or creating a collection."""
        # Reset the collection and reset the mock
        vector_db_service._collection = None
        mock_chromadb.get_or_create_collection.reset_mock()
        
        # Mock the embedding function
        with patch.object(vector_db_service, '_get_embedding_function', return_value=MagicMock()):
            # Call the method
            collection = vector_db_service.get_collection()
            
            # Verify the collection was retrieved
            assert collection is not None
            mock_chromadb.get_or_create_collection.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, vector_db_service):
        """Test generating embeddings from the embedding model."""
        embedding_function = MagicMock()
        embedding_function.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        with patch.object(vector_db_service, '_get_embedding_function', return_value=embedding_function):
            # Call the method
            result = await vector_db_service.generate_embeddings(["Test document 1", "Test document 2"])
            
            # Verify that the embedding function was called
            embedding_function.assert_called_once_with(["Test document 1", "Test document 2"])
            
            # Verify the result
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    @pytest.mark.asyncio
    async def test_add_document_chunks(self, vector_db_service):
        """Test adding document chunks to the vector database."""
        # Mock the get_collection method
        with patch.object(vector_db_service, 'get_collection') as mock_get_collection:
            mock_get_collection.return_value = vector_db_service._collection
            
            # Call the method
            result = await vector_db_service.add_document_chunks(
                documents=TEST_DOCUMENTS,
                metadatas=TEST_METADATA,
                ids=["doc1", "doc2"]
            )
            
            # Verify that the collection's add method was called
            vector_db_service._collection.add.assert_called_once()
            call_args = vector_db_service._collection.add.call_args[1]
            assert call_args["documents"] == TEST_DOCUMENTS
            assert call_args["metadatas"] == TEST_METADATA
            assert call_args["ids"] == ["doc1", "doc2"]
            
            # Verify the result
            assert result is True
    
    @pytest.mark.asyncio
    async def test_query_collection(self, vector_db_service):
        """Test querying the vector database."""
        # Mock the get_collection method
        with patch.object(vector_db_service, 'get_collection') as mock_get_collection:
            mock_get_collection.return_value = vector_db_service._collection
            
            # Call the method
            result = await vector_db_service.query_collection(
                query_texts=["Python developer with 5 years experience"],
                n_results=2,
                where={"doc_type": "job"}
            )
            
            # Verify that the collection's query method was called
            vector_db_service._collection.query.assert_called_once()
            call_args = vector_db_service._collection.query.call_args[1]
            assert call_args["query_texts"] == ["Python developer with 5 years experience"]
            assert call_args["n_results"] == 2
            assert call_args["where"] == {"doc_type": "job"}
            
            # Verify the result
            assert result == MOCK_QUERY_RESULTS 