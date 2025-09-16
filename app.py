import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from groq import Groq
import json, re, os
from dotenv import load_dotenv

load_dotenv()


client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def clean_json(raw_text: str):
    no_think = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", no_think, flags=re.DOTALL)
    return match.group(0).strip() if match else "{}"

def strip_think(text: str):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# --- Streamlit UI ---
st.title("📋 JobFit AI: Resume Evaluation and Cold Email Automation")

job_url = st.text_input("Job Post URL")
resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

# ---------- RUN ANALYSIS ----------
if st.button("Run Analysis"):
    if not job_url or not resume_file:
        st.warning("Please upload resume and job post")
    else:
        with st.spinner("🔍 Running ATS Analysis... Please wait"):
            # Load job
            job_docs = WebBaseLoader(web_paths=[job_url]).load()
            job_text = "\n".join([d.page_content for d in job_docs])

            # Load resume
            pdf = PdfReader(resume_file)
            content = "".join([p.extract_text() for p in pdf.pages])

            # Split and embed
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            resume_chunks = splitter.split_text(content)
            job_chunks = splitter.split_documents(job_docs)
            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            docs = [Document(page_content=c, metadata={"source":"resume"}) for c in resume_chunks] + \
                   [Document(page_content=d.page_content, metadata={"source":"job"}) for d in job_chunks]
            vs = Chroma.from_documents(docs, embedding=embed,persist_directory=None)
            
            
            retriever = vs.as_retriever()

            # ATS eval
            ats_prompt = f"""
You are an ATS evaluator. Compare this resume with the job description.

Job Description:
{job_text}

Resume:
{content}

Return JSON:
{{
  "ats_score": "<0-100 score based on how well skills & experience match>",
  "matched_skills": [],
  "missing_skills": [],
  "suggestions": [
    "Add more details about X project",
    "Mention Y skill from job description",
    "Include metrics or achievements in experience"
  ]
}}
"""
            resp = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[{"role": "user", "content": ats_prompt}]
            )
            ats_result = clean_json(resp.choices[0].message.content)

            # Store in session_state
            st.session_state["ats_result"] = ats_result
            st.session_state["retriever"] = retriever
            st.session_state["resume_text"] = content
            st.session_state["job_text"] = job_text

            # Extract company name
            comp_prompt = f"""
Extract ONLY the company name from this job description text:

{job_text}

Rules:
- Return only the company name (one word).
- Do NOT include any explanation or thinking.
- Do NOT include <think> tags.
- Output should look like: Speechify
"""
            c_resp = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[{"role": "user", "content": comp_prompt}]
            )
            company = c_resp.choices[0].message.content
            st.session_state["company"] = strip_think(company)

# ---------- SHOW ATS IF EXISTS ----------
if "ats_result" in st.session_state:
    ats_data = json.loads(st.session_state["ats_result"])

    st.subheader("📊 ATS Evaluation Result")

    ats_score = float(ats_data.get('ats_score', 0))  # default 0 if key missing

# Show metric
    st.metric("ATS Match Score", f"{ats_score}%")


    # --- Score with progress bar ---
    # st.metric("ATS Match Score", f"{ats_data['ats_score']}%")
    st.progress(ats_data['ats_score'] / 100)

    # --- Skills section ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Matched Skills")
        if ats_data["matched_skills"]:
            st.markdown("\n".join([f"- 🟢 **{s}**" for s in ats_data["matched_skills"]]))
        else:
            st.info("No matched skills found")

    with col2:
        st.markdown("### ⚠️ Missing Skills")
        if ats_data["missing_skills"]:
            st.markdown("\n".join([f"- 🔴 **{s}**" for s in ats_data["missing_skills"]]))
        else:
            st.success("No missing skills 🎉")

    # --- Suggestions section ---
    st.markdown("### 💡 Suggestions to Improve")
    if ats_data["suggestions"]:
        st.markdown("\n".join([f"- 💭 {s}" for s in ats_data["suggestions"]]))
    else:
        st.info("No suggestions provided.")

    st.markdown("---")
    st.subheader("📩 Cold Email Generator")

    # ---------- COLD EMAIL BUTTON ----------
    if st.button("Generate Cold Email"):
        with st.spinner("✍️ Writing your cold email..."):
            retriever = st.session_state["retriever"]
            content = st.session_state["resume_text"]
            job_text = st.session_state["job_text"]

            # Get context from vector DB
            query = "Summarize the candidate's best experience and skills for this job role"
            context_docs = retriever.get_relevant_documents(query)
            context_text = "\n\n".join([d.page_content for d in context_docs])
            template = """
You are drafting a professional cold email to apply for a job.

**Goal:** Write a concise, enthusiastic, confident cold email (150–200 words) tailored to the job.

**Instructions:**
- First, analyze the candidate's resume, the job description, and the given context.
- Identify: job title and company name from the job description.
- Create the **subject line in this exact format:** "Application for [Job Title] Role at [Company Name]"
- Then write the email body. Do NOT copy-paste resume — summarize it and show enthusiasm.
- Highlight top 2–3 skills or achievements that match the job.

**Tone:** Professional, enthusiastic, confident  
**Length:** 150–200 words

Candidate Resume:
{resume}

Job Description:
{job}

Context (from vector DB retrieval):
{context}

Output Format:
Subject: Application for <Job Title> Role at <Company Name>
Email:
<email body>
"""


           


            # Prompt template
#             template = """
# Write a professional, concise cold email to apply for the below job.

# Use the candidate's resume context and job post context. Mention key matching skills and show enthusiasm.

# **Tone:** Professional, enthusiastic, confident  
# **Length:** 150–200 words  
# **Do NOT copy-paste resume — summarize it.**  
# **Add a subject line too.**

# Candidate Resume:
# {resume}

# Job Description:
# {job}

# Context (from vector DB retrieval):
# {context}

# Email:
# """
           
            prompt = PromptTemplate(
                template=template,
                input_variables=["resume", "job", "context"]
            )

            # Generate email
            final_prompt = prompt.format(
                resume=content,
                job=job_text,
                context=context_text
            )

            resp = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[{"role": "user", "content": final_prompt}]
            )
            email_text = resp.choices[0].message.content.strip()

            # Try to split subject and body
            subject_line = ""
            body_text = email_text
            match = re.search(r"(Subject:.*?\n)([\s\S]*)", email_text, flags=re.IGNORECASE)
            if match:
                subject_line = match.group(1).strip()
                body_text = match.group(2).strip()

            # Show styled result
            st.markdown("### ✉️ Generated Cold Email")

            st.markdown(
                f"""
                <div style="background-color:#eef7ff; padding:20px; border-radius:10px; border:1px solid #cce5ff;">
                    <h4 style="margin-top:0;">📌 {subject_line or "Subject: Cold Job Application"}</h4>
                    <pre style="white-space:pre-wrap; font-size:15px; line-height:1.5;">{body_text}</pre>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Copy to clipboard
            copy_js = f"""
                <script>
                function copyToClipboard() {{
                    navigator.clipboard.writeText({json.dumps(email_text)});
                    alert('Copied to clipboard!');
                }}
                </script>
                <button onclick="copyToClipboard()" style="margin-top:10px; padding:8px 15px; background:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer;">
                    📋 Copy to Clipboard
                </button>
            """
            st.components.v1.html(copy_js, height=80)

    
