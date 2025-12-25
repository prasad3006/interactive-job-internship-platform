from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def skill_match(student_skills: str, job_skills: str) -> float:
    """
    Calculates skill match percentage between a student and a job/internship.

    Parameters:
    student_skills (str): Skills of the student (comma-separated)
    job_skills (str): Required skills for the job (comma-separated)

    Returns:
    float: Match percentage (0–100)
    """

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Convert skills into vectors
    vectors = vectorizer.fit_transform([student_skills, job_skills])

    # Calculate cosine similarity
    similarity_score = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    # Convert to percentage
    match_percentage = round(similarity_score * 100, 2)

    return match_percentage
