import streamlit as st
import ranking
import pandas as pd
import base64

st.set_page_config(page_title="ATS Resume Checker", layout="wide")

st.title("📄 AI-Powered ATS Resume Screening")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Enter Job Description")
    job_desc = st.text_area("Paste the job description here")

with col2:
    st.subheader("📂 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )

# 🔥 Function to create download link
def create_download_link(file):
    file.seek(0)
    b64 = base64.b64encode(file.read()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{file.name}">Download</a>'

if st.button("Check ATS Score") and uploaded_files and job_desc:
    results = []

    for uploaded_file in uploaded_files:
        resume_text = ranking.extract_text_from_pdf(uploaded_file)
        ats_score = ranking.bert_similarity(job_desc, resume_text) * 100

        results.append({
            "Resume Name": uploaded_file.name,
            "ATS Score": ats_score,
            "Download": create_download_link(uploaded_file)
        })

    # ✅ Sort descending
    results = sorted(results, key=lambda x: x["ATS Score"], reverse=True)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df["ATS Score"] = df["ATS Score"].map(lambda x: f"{x:.2f}%")

    st.subheader("📊 ATS Score Results")

    st.markdown("""
    <style>
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 15px;
    }
    .custom-table th {
        background-color: #262730;
        color: white;
        padding: 12px;
    }
    .custom-table td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    .custom-table tr:hover {
        background-color: #5c5858;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
    df.to_html(classes="custom-table", escape=False, index=False),
    unsafe_allow_html=True
    )