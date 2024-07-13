from PyPDF2 import PdfReader
from transformers import BertTokenizer
import os


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_pdf_content(document):
    # Print the document path for debugging
    print(f"Attempting to open: {document}")
    
    # Check if the file exists
    if not os.path.isfile(document):
        raise FileNotFoundError(f"No such file: '{document}'")
    
    # Read the PDF
    pdf_reader = PdfReader(document)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text




def get_chunks(text, max_length=512):
   
    tokens = tokenizer.encode(text, add_special_tokens=True)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return chunks

def decode_chunk(chunk):
    return tokenizer.decode(chunk)