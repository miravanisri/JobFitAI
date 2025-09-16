https://github.com/user-attachments/assets/64f50a8d-8730-49d6-9ad5-91c82a736ce6

# 📋 JobFit AI: Resume Evaluation & Cold Email Automation

**JobFit AI** is a Streamlit-based web application that helps job seekers analyze their resumes against job postings using ATS (Applicant Tracking System) scoring and generate professional, tailored cold emails for job applications. It leverages LangChain, Groq API, and vector embeddings for intelligent document analysis.

---

## 🌟 Features

1. **ATS Resume Evaluation**
   - Upload your resume (PDF) and provide a job posting URL.
   - Get an ATS score (0–100%) indicating how well your resume matches the job.
   - View **matched skills**, **missing skills**, and **actionable suggestions** to improve your resume.

2. **Cold Email Generator**
   - Generate professional, concise, and enthusiastic cold emails based on your resume and job description.
   - Includes automatically generated **subject lines** tailored to the role and company.
   - Copy email content to clipboard with one click.

3. **Interactive & User-Friendly UI**
   - Beautiful progress bars and skill comparison layout.
   - Styled output for email and ATS suggestions.

---

## 🛠️ Tech Stack

- **Backend & AI:** Python, LangChain, Groq API, HuggingFace Embeddings, ChromaDB  
- **Frontend:** Streamlit  
- **Document Processing:** PyPDF2, Python-docx  
- **Web Scraping:** BeautifulSoup4 (`bs4`)  
- **Deployment Tools:** Pyngrok (for local tunneling/testing)  

---

## ⚡ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/jobfit-ai.git
cd jobfit-ai
````

2. Create a virtual environment and activate it:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_HF_token_api_key_here
LANGSMITH_API_KEY=your_Langsmith_api_key_here

```

5. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🚀 Usage

1. Open the app in your browser (default `http://localhost:8501`).
2. Enter the **Job Posting URL**.
3. Upload your **Resume PDF**.
4. Click **Run Analysis** to get your ATS score and skill suggestions.
5. Click **Generate Cold Email** to create a personalized application email.




## 📄 Example Output

**ATS Score:** 78%
**Matched Skills:** TypeScript, Node.js, Docker, AWS
**Missing Skills:** Kubernetes, GCP, Payment Systems
**Suggestions:** Add metrics for API optimization, include cloud projects, etc.

**Generated Cold Email:**


* Personalized email body summarizing skills & experience

---

## 📜 License

MIT License © 2025




