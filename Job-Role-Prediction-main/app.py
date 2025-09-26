import streamlit as st
import pickle
import re
import nltk
from PyPDF2 import PdfReader

# --- SETUP: Download NLTK resources and load models ---
@st.cache_resource
def load_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    clf = pickle.load(open('clf.pkl', 'rb'))
    tfidfd = pickle.load(open('tfidf.pkl', 'rb'))
    return clf, tfidfd

clf, tfidfd = load_resources()

# --- CATEGORY MAPPING ---
CATEGORY_MAPPING = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "SDE",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain",
    10: "ETL Developer", 18: "Operations Manager", 6: "Data Science", 22: "Sales",
    16: "Mechanical Engineer", 1: "Arts", 7: "Database", 11: "Electrical Engineering",
    14: "Health and fitness", 19: "PMO", 4: "Business Analyst", 9: "DotNet Developer",
    2: "Automation Testing", 17: "Network Security Engineer", 21: "SAP Developer",
    5: "Civil Engineer", 0: "Advocate",
}

# --- HELPER FUNCTIONS ---
def clean_resume(resume_text):
    """Clean the resume text by removing URLs, special characters, etc."""
    clean_text = re.sub(r'http\S+\s*', ' ', resume_text)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+', '', clean_text)
    clean_text = re.sub(r'@\S+', '  ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text.strip()

def extract_text_from_pdf(file):
    """Extract text from an uploaded PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# --- MAIN APPLICATION ---
def main():
    # --- PAGE CONFIGURATION ---
    st.set_page_config(
        page_title="Resume Screening AI",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # --- PURPLE THEME CSS ---
    st.markdown("""
    <style>
        /* App background */
        .stApp {
            background: linear-gradient(135deg, #ede7f6, #d1c4e9);
            color: #2c0341;
            font-family: 'Segoe UI', Tahoma, sans-serif;
        }

        /* Headers */
        h1.title {
            color: #6a1b9a;
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            padding-top: 20px;
            margin-bottom: 0px;
        }
        p.subheader {
            color: #4a148c;
            text-align: center;
            font-size: 18px;
            margin-bottom: 20px;
        }

        /* File uploader container */
        div.stFileUploader {
            background-color: #f3e5f5;
            border: 2px dashed #7b1fa2;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
        }
        div.stFileUploader label {
            color: #4a148c;
            font-weight: 600;
        }

        /* Buttons */
        button.stButton>button {
            background-color: #6a1b9a;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.5em 1.5em;
        }
        button.stButton>button:hover {
            background-color: #8e24aa;
        }

        /* Result container */
        .result-container {
            background-color: #f3e5f5;
            border-radius: 16px;
            padding: 30px;
            margin-top: 20px;
            border: 1px solid #ce93d8;
            box-shadow: 0 12px 24px rgba(0,0,0,0.08);
            text-align: center;
        }
        .prediction {
            font-size: 32px;
            font-weight: 700;
            color: #4a148c;
            margin-top: 10px;
        }

        /* Info box styling */
        div[data-baseweb="alert"] {
            background-color: #e1bee7 !important;
            border-radius: 12px !important;
            border: 1px solid #ba68c8 !important;
            color: #4a148c !important;
        }

        /* Text area for extracted snippet */
        textarea {
            background-color: #f8f0fa;
            border: 1px solid #ce93d8;
            border-radius: 8px;
            padding: 10px;
            font-family: 'Segoe UI', sans-serif;
            color: #2c0341;
        }

        /* Description text */
        .stMarkdown p {
            color: #4a148c;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- HEADER ---
    st.markdown('<h1 class="title">🤖 AI-Powered Resume Screener</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload a resume to automatically classify it into a relevant job category.</p>', unsafe_allow_html=True)
    st.divider()

    # --- MAIN LAYOUT (2 columns) ---
    col1, col2 = st.columns([2, 1.5], gap="large")

    with col1:
        st.header("📄 Upload Your Resume")
        uploaded_file = st.file_uploader(
            "Choose a file (.txt or .pdf)", 
            type=["txt", "pdf"],
            help="Upload the resume you want the AI to analyze."
        )

        if uploaded_file is not None:
            with st.spinner("🧠 Analyzing resume... Please wait."):
                try:
                    # --- File Processing and Prediction ---
                    if uploaded_file.type == "application/pdf":
                        resume_text = extract_text_from_pdf(uploaded_file)
                    else:  # For .txt files
                        resume_text = uploaded_file.read().decode(errors="ignore")

                    if not resume_text.strip():
                        st.error("❌ Could not extract text. The file might be empty or corrupted.")
                        return

                    cleaned_resume = clean_resume(resume_text)
                    input_features = tfidfd.transform([cleaned_resume])
                    prediction_id = clf.predict(input_features)[0]
                    category_name = CATEGORY_MAPPING.get(prediction_id, "Unknown")

                    # --- Display Result ---
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.success("✅ Analysis Complete!")
                    st.markdown("<p>The resume is most likely for the role of:</p>", unsafe_allow_html=True)
                    st.markdown(f'<p class="prediction">{category_name}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # --- Show a snippet of the extracted text ---
                    with st.expander("Show Extracted Text Snippet"):
                        st.text_area("", resume_text[:600] + "...", height=150, disabled=True)

                except Exception as e:
                    st.error(f"⚠️ An unexpected error occurred: {e}")

    with col2:
        st.header("✨ How It Works")
        st.markdown("""
        This tool uses a Machine Learning model to analyze the text of a resume and predict the most suitable job category.
        
        1.  **Upload:** Provide a resume in `.pdf` or `.txt` format.
        2.  **Text Extraction:** The system extracts the raw text from the document.
        3.  **Text Cleaning:** Irrelevant information like URLs and special characters are removed.
        4.  **Prediction:** A pre-trained model analyzes keywords to predict the best-fit job role.
        """)
        st.info("Disclaimer: This model's prediction is based on the patterns from its training data. It is intended as a helpful suggestion, not a definitive classification.", icon="ℹ️")

if __name__ == "__main__":
    main()
