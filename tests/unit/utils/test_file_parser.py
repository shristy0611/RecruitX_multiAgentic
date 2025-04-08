import os
import sys
import pytest
from unittest.mock import patch, MagicMock, Mock, ANY
import io

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from recruitx_app.utils.file_parser import (
    extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt, 
    extract_text_from_file, clean_text, extract_text_from_pdf_with_layout,
    extract_text_with_ocr, extract_text_from_rtf, extract_text_from_csv,
    extract_text_from_html
)


class TestFileParser:
    """Test class for file parsing utilities."""
    
    def test_clean_text(self):
        """Test the text cleaning function."""
        # Test multiple spaces
        assert clean_text("This   has   extra    spaces") == "This has extra spaces"
        
        # Test paragraph breaks
        assert clean_text("First sentence. Second sentence") == "First sentence.\nSecond sentence"
        
        # Test empty input
        assert clean_text("") == ""
        assert clean_text(None) == ""
        
        # Test non-printable characters
        assert clean_text("Text with\x00non-printable\x1Fchars") == "Text withnon-printablechars"
    
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
    
    def test_extract_text_from_txt_with_multiple_encodings(self):
        """Test handling of different encodings in TXT files."""
        # Create text in various encodings
        utf8_text = "UTF-8 encoded text with special chars: åéîøü".encode('utf-8')
        latin1_text = "Latin-1 encoded text with special chars: åéîøü".encode('latin-1')
        utf16_text = "UTF-16 encoded text with special chars: åéîøü".encode('utf-16')
        
        # Test each encoding
        assert "UTF-8 encoded text" in extract_text_from_txt(utf8_text)
        assert "Latin-1 encoded text" in extract_text_from_txt(latin1_text)
        assert "UTF-16 encoded text" in extract_text_from_txt(utf16_text)
    
    @patch('recruitx_app.utils.file_parser.extract_text_from_pdf_with_layout')
    @patch('recruitx_app.utils.file_parser.PdfReader')
    def test_extract_text_from_pdf_with_layout_first(self, mock_pdf_reader, mock_layout_extract):
        """Test that layout-aware extraction is attempted first."""
        # Setup layout extract to return text
        mock_layout_extract.return_value = "Layout-aware extracted text"
        
        # Extract text - should use layout-aware method and not fall back to PdfReader
        result = extract_text_from_pdf(b'mock_pdf_content')
        
        # Verify layout-aware extraction was called
        mock_layout_extract.assert_called_once_with(b'mock_pdf_content')
        
        # Verify PdfReader was not called (no fallback needed)
        mock_pdf_reader.assert_not_called()
        
        # Verify the result
        assert result == "Layout-aware extracted text"
    
    @patch('recruitx_app.utils.file_parser.extract_text_from_pdf_with_layout')
    @patch('recruitx_app.utils.file_parser.PdfReader')
    def test_extract_text_from_pdf_fallback(self, mock_pdf_reader, mock_layout_extract):
        """Test PDF extraction fallback when layout-aware extraction fails."""
        # Setup layout extract to return empty text (failure)
        mock_layout_extract.return_value = ""
        
        # Create mock PDF pages with text
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        # Configure the mock PdfReader
        mock_pdf_instance = MagicMock()
        mock_pdf_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_pdf_instance
        
        # Extract text - should fall back to PdfReader
        result = extract_text_from_pdf(b'mock_pdf_content')
        
        # Verify layout-aware extraction was called first
        mock_layout_extract.assert_called_once_with(b'mock_pdf_content')
        
        # Verify PdfReader was called as fallback
        mock_pdf_reader.assert_called_once()
        
        # Verify the result
        assert "Page 1 content" in result
        assert "Page 2 content" in result
    
    @patch('recruitx_app.utils.file_parser.fitz')
    def test_extract_text_from_pdf_with_layout(self, mock_fitz):
        """Test extracting text from a PDF with layout awareness using PyMuPDF."""
        # Configure mock document and pages
        mock_doc = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 with layout"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 with layout"
        
        # Set up the mock chain
        mock_fitz.open.return_value = mock_doc
        mock_doc.load_page.side_effect = [mock_page1, mock_page2]
        mock_doc.__len__.return_value = 2
        
        # Test with mocked temporary file
        with patch('recruitx_app.utils.file_parser.tempfile.NamedTemporaryFile') as mock_temp:
            # Set up the mock temporary file
            mock_temp_instance = MagicMock()
            mock_temp_instance.name = "/tmp/mock_temp_file.pdf"
            mock_temp.return_value.__enter__.return_value = mock_temp_instance
            
            # Call the function
            result = extract_text_from_pdf_with_layout(b'mock_pdf_content')
            
            # Verify temp file was created and written to
            mock_temp_instance.write.assert_called_once_with(b'mock_pdf_content')
            
            # Verify document was opened
            mock_fitz.open.assert_called_once_with("/tmp/mock_temp_file.pdf")
            
            # Verify pages were loaded and text was extracted
            assert mock_doc.load_page.call_count == 2
            assert mock_page1.get_text.call_count == 1
            assert mock_page2.get_text.call_count == 1
            
            # Verify document was closed
            assert mock_doc.close.call_count == 1
            
            # Verify the result
            assert "Page 1 with layout" in result
            assert "Page 2 with layout" in result
    
    @patch('recruitx_app.utils.file_parser.pytesseract')
    @patch('recruitx_app.utils.file_parser.Image')
    @patch('recruitx_app.utils.file_parser.fitz')
    @patch('recruitx_app.utils.file_parser.os.path.exists')
    @patch('recruitx_app.utils.file_parser.os.unlink')
    def test_extract_text_with_ocr(self, mock_unlink, mock_exists, mock_fitz, mock_image, mock_pytesseract):
        """Test OCR text extraction from scanned documents."""
        # Mock exists for temp file cleanup
        mock_exists.return_value = True
        
        # Configure mock document and pages
        mock_doc = MagicMock()
        mock_page1 = MagicMock()
        mock_pixmap1 = MagicMock()
        mock_page2 = MagicMock()
        mock_pixmap2 = MagicMock()
        
        # Set up the mock chain
        mock_fitz.open.return_value = mock_doc
        mock_doc.load_page.side_effect = [mock_page1, mock_page2]
        mock_page1.get_pixmap.return_value = mock_pixmap1
        mock_page2.get_pixmap.return_value = mock_pixmap2
        mock_doc.__len__.return_value = 2
        
        # Mock OCR results
        mock_pytesseract.image_to_string.side_effect = ["OCR text page 1", "OCR text page 2"]
        
        # Mock image opening
        mock_image.open.side_effect = [MagicMock(), MagicMock()]
        
        # Test with mocked temporary file
        with patch('recruitx_app.utils.file_parser.tempfile.NamedTemporaryFile') as mock_temp:
            # Set up the mock temporary file
            mock_temp_instance = MagicMock()
            mock_temp_instance.name = "/tmp/mock_temp_file.pdf"
            mock_temp.return_value.__enter__.return_value = mock_temp_instance
            
            # Set OCR_AVAILABLE to True during test
            with patch('recruitx_app.utils.file_parser.OCR_AVAILABLE', True):
                # Call the function
                result = extract_text_with_ocr(b'mock_pdf_content')
                
                # Verify temp file was created and written to
                mock_temp_instance.write.assert_called_once_with(b'mock_pdf_content')
                
                # Verify document was opened and processed
                mock_fitz.open.assert_called_once_with("/tmp/mock_temp_file.pdf")
                assert mock_doc.load_page.call_count == 2
                
                # Verify pixmaps were created and saved
                assert mock_page1.get_pixmap.call_count == 1
                assert mock_page2.get_pixmap.call_count == 1
                assert mock_pixmap1.save.call_count == 1
                assert mock_pixmap2.save.call_count == 1
                
                # Verify OCR was performed
                assert mock_pytesseract.image_to_string.call_count == 2
                
                # Verify temp files were cleaned up
                assert mock_unlink.call_count >= 1
                
                # Verify the result
                assert "OCR text page 1" in result
                assert "OCR text page 2" in result
    
    def test_ocr_not_available(self):
        """Test handling when OCR is not available."""
        with patch('recruitx_app.utils.file_parser.OCR_AVAILABLE', False):
            result = extract_text_with_ocr(b'mock_content')
            assert result == ""
    
    @patch('recruitx_app.utils.file_parser.PdfReader')
    def test_extract_text_from_pdf_with_error(self, mock_pdf_reader):
        """Test error handling when extracting text from a PDF file."""
        # Mock layout extraction to fail
        with patch('recruitx_app.utils.file_parser.extract_text_from_pdf_with_layout', return_value=""):
            # Make the PdfReader raise an exception
            mock_pdf_reader.side_effect = Exception("PDF extraction error")
            
            # Mock OCR not available
            with patch('recruitx_app.utils.file_parser.OCR_AVAILABLE', False):
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
        
        # Create mock tables
        mock_cell1 = MagicMock()
        mock_cell1.text = "Cell 1"
        mock_cell2 = MagicMock()
        mock_cell2.text = "Cell 2"
        mock_row = MagicMock()
        mock_row.cells = [mock_cell1, mock_cell2]
        mock_table = MagicMock()
        mock_table.rows = [mock_row]
        
        # Configure the mock Document
        mock_doc_instance = MagicMock()
        mock_doc_instance.paragraphs = [mock_para1, mock_para2]
        mock_doc_instance.tables = [mock_table]
        mock_document.return_value = mock_doc_instance
        
        # Test the function
        result = extract_text_from_docx(b'mock_docx_content')
        
        # Verify Document was created with BytesIO object
        mock_document.assert_called_once()
        assert isinstance(mock_document.call_args[0][0], io.BytesIO)
        
        # Verify text extraction from paragraphs and tables
        assert "Paragraph 1 content" in result
        assert "Paragraph 2 content" in result
        assert "Cell 1 | Cell 2" in result
    
    @patch('recruitx_app.utils.file_parser.Document')
    def test_extract_text_from_docx_with_error(self, mock_document):
        """Test error handling when extracting text from a DOCX file."""
        # Make the Document raise an exception
        mock_document.side_effect = Exception("DOCX extraction error")
        
        # Test the function
        result = extract_text_from_docx(b'mock_docx_content')
        
        # Verify error handling
        assert result == ""
    
    def test_extract_text_from_rtf(self):
        """Test extracting text from an RTF file."""
        # Sample RTF content
        rtf_content = r'{\rtf1\ansi\ansicpg1252\cocoartf2706 Sample RTF content with \b bold \b0 formatting.}'.encode('ascii')
        
        # Test the function
        result = extract_text_from_rtf(rtf_content)
        
        # Verify RTF commands are removed
        assert "Sample RTF content with  bold  formatting" in result
    
    def test_extract_text_from_csv(self):
        """Test extracting text from a CSV file."""
        # Sample CSV content
        csv_content = 'Name,Email,Phone\nJohn Doe,john@example.com,555-1234\nJane Smith,jane@example.com,555-5678'.encode('utf-8')
        
        # Test the function
        result = extract_text_from_csv(csv_content)
        
        # Verify CSV conversion
        assert "Name | Email | Phone" in result
        assert "John Doe | john@example.com | 555-1234" in result
        assert "Jane Smith | jane@example.com | 555-5678" in result
    
    @patch('recruitx_app.utils.file_parser.html2text.HTML2Text')
    def test_extract_text_from_html(self, mock_html2text_class):
        """Test extracting text from an HTML file."""
        # Configure mock HTML2Text instance
        mock_html2text = MagicMock()
        mock_html2text.handle.return_value = "Extracted HTML Content"
        mock_html2text_class.return_value = mock_html2text
        
        # Sample HTML content
        html_content = '<html><body><h1>Sample Heading</h1><p>Sample paragraph</p></body></html>'.encode('utf-8')
        
        # Test the function
        result = extract_text_from_html(html_content)
        
        # Verify HTML2Text was configured correctly
        assert mock_html2text.ignore_links == True
        assert mock_html2text.ignore_images == True
        
        # Verify handle was called with decoded HTML
        mock_html2text.handle.assert_called_once()
        
        # Verify the result
        assert result == "Extracted HTML Content"
    
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
    
    def test_extract_text_from_file_doc(self):
        """Test extracting text from a DOC file using the general method."""
        with patch('recruitx_app.utils.file_parser.extract_text_from_docx', return_value="DOC content") as mock_docx:
            result = extract_text_from_file(b'content', "sample.doc")
            
            mock_docx.assert_called_once_with(b'content')
            assert result == "DOC content"
    
    def test_extract_text_from_file_txt(self):
        """Test extracting text from a TXT file using the general method."""
        with patch('recruitx_app.utils.file_parser.extract_text_from_txt', return_value="TXT content") as mock_txt:
            result = extract_text_from_file(b'content', "sample.txt")
            
            mock_txt.assert_called_once_with(b'content')
            assert result == "TXT content"
    
    def test_extract_text_from_file_rtf(self):
        """Test extracting text from an RTF file using the general method."""
        with patch('recruitx_app.utils.file_parser.extract_text_from_rtf', return_value="RTF content") as mock_rtf:
            result = extract_text_from_file(b'content', "sample.rtf")
            
            mock_rtf.assert_called_once_with(b'content')
            assert result == "RTF content"
    
    def test_extract_text_from_file_csv(self):
        """Test extracting text from a CSV file using the general method."""
        with patch('recruitx_app.utils.file_parser.extract_text_from_csv', return_value="CSV content") as mock_csv:
            result = extract_text_from_file(b'content', "sample.csv")
            
            mock_csv.assert_called_once_with(b'content')
            assert result == "CSV content"
    
    def test_extract_text_from_file_html(self):
        """Test extracting text from an HTML file using the general method."""
        with patch('recruitx_app.utils.file_parser.extract_text_from_html', return_value="HTML content") as mock_html:
            result = extract_text_from_file(b'content', "sample.html")
            
            mock_html.assert_called_once_with(b'content')
            assert result == "HTML content"
    
    def test_extract_text_from_file_image_with_ocr(self):
        """Test extracting text from an image file using OCR."""
        with patch('recruitx_app.utils.file_parser.OCR_AVAILABLE', True):
            with patch('recruitx_app.utils.file_parser.extract_text_with_ocr', return_value="OCR content") as mock_ocr:
                result = extract_text_from_file(b'content', "sample.jpg")
                
                mock_ocr.assert_called_once_with(b'content')
                assert result == "OCR content"
    
    def test_extract_text_from_file_image_without_ocr(self):
        """Test handling of image files when OCR is not available."""
        with patch('recruitx_app.utils.file_parser.OCR_AVAILABLE', False):
            result = extract_text_from_file(b'content', "sample.jpg")
            assert result is None
    
    def test_extract_text_from_file_empty_content(self):
        """Test handling of empty file content."""
        result = extract_text_from_file(None, "sample.txt")
        assert result is None
        
        result = extract_text_from_file(b'', "sample.txt")
        assert result is None
    
    def test_extract_text_from_file_unsupported(self):
        """Test handling of unsupported file types."""
        result = extract_text_from_file(b'content', "sample.unsupported")
        assert result is None
    
    def test_extract_text_from_file_with_text_cleanup(self):
        """Test that text cleaning is applied to extracted text."""
        with patch('recruitx_app.utils.file_parser.extract_text_from_txt', return_value="Text   with   extra  spaces") as mock_txt:
            with patch('recruitx_app.utils.file_parser.clean_text', return_value="Text with extra spaces") as mock_clean:
                result = extract_text_from_file(b'content', "sample.txt")
                
                mock_txt.assert_called_once_with(b'content')
                mock_clean.assert_called_once_with("Text   with   extra  spaces")
                assert result == "Text with extra spaces" 