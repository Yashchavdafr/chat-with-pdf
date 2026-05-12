"""
loader.py — PDF text extraction module.

Reads all pages from a PDF file using pdfplumber and returns
the full document text as a single cleaned string.
"""

import pdfplumber
import re


def extract_text(pdf_path: str) -> str:
    """
    Extract and clean text from a PDF file.

    Opens every page with pdfplumber, concatenates the raw text,
    strips excessive whitespace and blank lines, and returns the
    result as a single string.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A single string containing all readable text from the PDF,
        with consecutive blank lines collapsed and leading/trailing
        whitespace removed from each line.

    Raises:
        FileNotFoundError: If the PDF file does not exist at pdf_path.
        pdfplumber.exceptions.PDFSyntaxError: If the file is not a
            valid or readable PDF.
    """
    pages_text: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text()
            if raw:
                pages_text.append(raw)

    full_text = "\n".join(pages_text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in full_text.splitlines()]

    # Remove runs of more than one consecutive blank line
    cleaned_lines: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        cleaned_lines.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned_lines).strip()
