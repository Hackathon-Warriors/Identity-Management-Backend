from PyPDF2 import PdfReader
import re

def is_valid_pdf(pdf_path, password=None, search_words=None) -> bool:
    """
    Check PDF metadata for specific keywords.
    
    Args:
        pdf_path (str): Path to the PDF file
        password (str, optional): PDF password if encrypted
        search_words (list, optional): List of keywords to search for
        
    Returns:
        bool: True if any search word is found in metadata, False otherwise
    """
    if search_words is None:
        search_words = [
            "scan", "oken", "online2pdf", "zamzar", "ios", "microsoft",
            "office lens", "ilovepdf", "macos", "pdfmaker", "adobe reader",
            "adobe acrobat", "image", "print to pdf", "sejda", "xodo",
            "foxit", "pdf-xchange", "android", "canon", "vFlat"
        ]
    
    try:
        # Open PDF with password if provided
        try:
            reader = PdfReader(pdf_path)
        except Exception as e:
            # If encryption error, try with password
            if 'encrypted' in str(e).lower() and password:
                reader = PdfReader(pdf_path, password=password)
            else:
                raise e
        
        # Get metadata
        metadata = reader.metadata
        print(f"pdf metadata: {metadata}")
        
        if metadata:
            # Convert metadata to string and make it lowercase for case-insensitive search
            metadata_str = str(metadata).lower()
            
            # Check each search word
            for word in search_words:
                word = word.lower()
                # Use word boundaries in regex to avoid partial matches
                pattern = rf'\b{re.escape(word)}\b'
                if re.search(pattern, metadata_str):
                    return True
                    
        return False
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    # Example with password-protected PDF
    pdf_path = "/Users/divyanshnew/Downloads/cbc7c968-aa44-4e54-9324-ef98a80f221a_unlckd_rdct.pdf"
    pdf_password = ""  # Optional
    
    result = is_valid_pdf(pdf_path, password=pdf_password)
    print(f"Found matching keywords: {result}")