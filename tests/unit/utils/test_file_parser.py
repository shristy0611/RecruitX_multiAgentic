import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import io

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from recruitx_app.utils.file_parser import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt, extract_text_from_file


class TestFileParser:
    """Test class for file parsing utilities."""
    
    def test_extract_text_from_txt(self):
        """Test extracting text from a TXT file."""
        # Sample text content
        sample_content = "This is a sample text file.\nIt has multiple lines.\nTesting text extraction."
        
        # Convert to bytes (as if it was read from a file)
        txt_content = sample_content.encode('utf-8')
        
        # Extract text
        result = extract_text_from_txt(txt_content)
        
        # Verify the result
        assert result == sample_content
        assert "sample text file" in result
        assert "multiple lines" in result
    
    def test_extract_text_from_txt_with_encoding_error(self):
        """Test handling of encoding errors in TXT files."""
        # Instead of trying to patch bytes.decode, let's test the behavior directly
        # Create a byte string that will cause a UnicodeDecodeError with utf-8
        invalid_bytes = b'\xff\xfe\x00\x00' # Invalid UTF-8 bytes
        
        # The function should try latin-1 as fallback which can decode any byte
        result = extract_text_from_txt(invalid_bytes)
        
        # Latin-1 will decode these bytes to something, so result should not be empty
        assert result != ""
    
    @patch('recruitx_app.utils.file_parser.PdfReader')
    def test_extract_text_from_pdf(self, mock_pdf_reader):
        """Test extracting text from a PDF file."""
        # Create mock PDF pages with text
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        # Configure the mock PdfReader
        mock_pdf_instance = MagicMock()
        mock_pdf_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_pdf_instance
        
        # Test the function
        result = extract_text_from_pdf(b'mock_pdf_content')
        
        # Verify PdfReader was created with BytesIO object
        mock_pdf_reader.assert_called_once()
        assert isinstance(mock_pdf_reader.call_args[0][0], io.BytesIO)
        
        # Verify text extraction from pages
        assert "Page 1 content" in result
        assert "Page 2 content" in result
    
    @patch('recruitx_app.utils.file_parser.PdfReader')
    def test_extract_text_from_pdf_with_error(self, mock_pdf_reader):
        """Test error handling when extracting text from a PDF file."""
        # Make the PdfReader raise an exception
        mock_pdf_reader.side_effect = Exception("PDF extraction error")
        
        # Test the function
        result = extract_text_from_pdf(b'mock_pdf_content')
        
        # Verify error handling
        assert result == ""
    
    @patch('recruitx_app.utils.file_parser.Document')
    def test_extract_text_from_docx(self, mock_document):
        """Test extracting text from a DOCX file."""
        # Create mock paragraphs
        mock_para1 = MagicMock()
        mock_para1.text = "Paragraph 1 content"
        
        mock_para2 = MagicMock()
        mock_para2.text = "Paragraph 2 content"
        
        # Configure the mock Document
        mock_doc_instance = MagicMock()
        mock_doc_instance.paragraphs = [mock_para1, mock_para2]
        mock_document.return_value = mock_doc_instance
        
        # Test the function
        result = extract_text_from_docx(b'mock_docx_content')
        
        # Verify Document was created with BytesIO object
        mock_document.assert_called_once()
        assert isinstance(mock_document.call_args[0][0], io.BytesIO)
        
        # Verify text extraction from paragraphs
        assert "Paragraph 1 content" in result
        assert "Paragraph 2 content" in result
    
    @patch('recruitx_app.utils.file_parser.Document')
    def test_extract_text_from_docx_with_error(self, mock_document):
        """Test error handling when extracting text from a DOCX file."""
        # Make the Document raise an exception
        mock_document.side_effect = Exception("DOCX extraction error")
        
        # Test the function
        result = extract_text_from_docx(b'mock_docx_content')
        
        # Verify error handling
        assert result == ""
    
    def test_extract_text_from_file_pdf(self):
        """Test extracting text from a PDF file using the general method."""
        with patch('recruitx_app.utils.file_parser.extract_text_from_pdf', return_value="PDF content") as mock_pdf:
            result = extract_text_from_file(b'content', "sample.pdf")
            
            mock_pdf.assert_called_once_with(b'content')
            assert result == "PDF content"
    
    def test_extract_text_from_file_docx(self):
        """Test extracting text from a DOCX file using the general method."""
        with patch('recruitx_app.utils.file_parser.extract_text_from_docx', return_value="DOCX content") as mock_docx:
            result = extract_text_from_file(b'content', "sample.docx")
            
            mock_docx.assert_called_once_with(b'content')
            assert result == "DOCX content"
    
    def test_extract_text_from_file_txt(self):
        """Test extracting text from a TXT file using the general method."""
        with patch('recruitx_app.utils.file_parser.extract_text_from_txt', return_value="TXT content") as mock_txt:
            result = extract_text_from_file(b'content', "sample.txt")
            
            mock_txt.assert_called_once_with(b'content')
            assert result == "TXT content"
    
    def test_extract_text_from_file_unsupported(self):
        """Test handling of unsupported file types."""
        result = extract_text_from_file(b'content', "sample.unsupported")
        
        assert result is None 