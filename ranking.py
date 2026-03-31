from transformers import BertTokenizer, BertModel
import torch
import PyPDF2
import io

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text() + "\n"
    return text

def get_bert_embedding(text):
    """Generates BERT embeddings for a given text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Get sentence embedding

def bert_similarity(job_desc, resume_text):
    """Computes similarity score between job description and resume using BERT."""
    job_embedding = get_bert_embedding(job_desc)
    resume_embedding = get_bert_embedding(resume_text)
    similarity_score = torch.cosine_similarity(job_embedding, resume_embedding).item()
    return similarity_score
