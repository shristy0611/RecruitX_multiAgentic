import os
import io
from typing import Optional
from PyPDF2 import PdfReader
from docx import Document

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_content: The binary content of the PDF file
        
    Returns:
        The extracted text as a string
    """
    try:
        # Create a PDF reader object
        pdf_file = io.BytesIO(pdf_content)
        pdf = PdfReader(pdf_file)
        
        # Extract text from each page
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_content: bytes) -> str:
    """
    Extract text from a DOCX file.
    
    Args:
        docx_content: The binary content of the DOCX file
        
    Returns:
        The extracted text as a string
    """
    try:
        # Create a document object
        docx_file = io.BytesIO(docx_content)
        doc = Document(docx_file)
        
        # Extract text from the document
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_txt(txt_content: bytes) -> str:
    """
    Extract text from a TXT file.
    
    Args:
        txt_content: The binary content of the TXT file
        
    Returns:
        The extracted text as a string
    """
    try:
        # Decode the bytes to string
        return txt_content.decode("utf-8")
    except UnicodeDecodeError:
        # Try with another encoding if UTF-8 fails
        try:
            return txt_content.decode("latin-1")
        except Exception as e:
            print(f"Error decoding text file: {e}")
            return ""

def extract_text_from_file(file_content: bytes, filename: str) -> Optional[str]:
    """
    Extract text from a file based on its extension.
    
    Args:
        file_content: The binary content of the file
        filename: The name of the file (with extension)
        
    Returns:
        The extracted text as a string, or None if the file type is not supported
    """
    # Get the file extension
    _, extension = os.path.splitext(filename)
    extension = extension.lower()
    
    # Extract text based on file extension
    if extension == ".pdf":
        return extract_text_from_pdf(file_content)
    elif extension == ".docx":
        return extract_text_from_docx(file_content)
    elif extension == ".txt":
        return extract_text_from_txt(file_content)
    else:
        print(f"Unsupported file type: {extension}")
        return None 