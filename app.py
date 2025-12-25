import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Campus2Career",
    page_icon="🎓",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color:#1f77b4;'>
    🎓 Campus2Career
    </h1>
    <h4 style='text-align: center;'>
    Interactive Job & Internship Platform (Smart Education)
    </h4>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- LOAD DATA ----------------
students = pd.read_csv("data/students.csv")
jobs = pd.read_csv("data/jobs.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.header("👤 Student Login (Demo)")
student_name = st.sidebar.selectbox(
    "Select Student",
    students["name"].unique()
)

student = students[students["name"] == student_name].iloc[0]

# ---------------- STUDENT PROFILE ----------------
st.subheader("👨‍🎓 Student Profile")

col1, col2, col3 = st.columns(3)

col1.metric("Name", student["name"])
col2.metric("CGPA", student["cgpa"])
col3.metric("Degree", student["degree"])

st.markdown(f"**Skills:** {student['skills']}")

st.divider()

# ---------------- SKILL MATCH FUNCTION ----------------
def skill_match(student_skills, job_skills):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([student_skills, job_skills])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

# ---------------- JOB RECOMMENDATIONS ----------------
st.subheader("💼 Recommended Jobs & Internships")

for _, job in jobs.iterrows():
    match_score = skill_match(student["skills"], job["required_skills"])
    eligible = student["cgpa"] >= job["min_cgpa"]

    with st.container():
        st.markdown(
            f"""
            <div style="
                border:1px solid #ddd;
                padding:15px;
                border-radius:10px;
                margin-bottom:15px;
            ">
            <h4>{job['role']} ({job['type']})</h4>
            <p><b>Required Skills:</b> {job['required_skills']}</p>
            <p><b>Match Score:</b> {match_score}%</p>
            <p><b>Eligibility:</b> {"✅ Eligible" if eligible else "❌ Not Eligible"}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center;'>Powered by AI • Built for GDG TechSprint 🚀</p>",
    unsafe_allow_html=True
)
