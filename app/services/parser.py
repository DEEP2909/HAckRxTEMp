import os
from typing import List
from PyPDF2 import PdfReader
import docx
import html2text

def parse_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages)

def parse_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)

def parse_html(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return html2text.html2text(f.read())

def parse_email(file_path: str) -> str:
    import email
    with open(file_path, 'r') as f:
        msg = email.message_from_file(f)
    return msg.get_payload()

def parse_document(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        return parse_pdf(file_path)
    elif file_path.endswith(".docx"):
        return parse_docx(file_path)
    elif file_path.endswith(".html"):
        return parse_html(file_path)
    elif file_path.endswith(".eml"):
        return parse_email(file_path)
    elif file_path.endswith(".txt"):
        return open(file_path).read()
    raise ValueError("Unsupported file format")
