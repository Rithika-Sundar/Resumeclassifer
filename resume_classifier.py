import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define skills to extract
skill_keywords = [
    'python', 'java', 'react', 'node', 'html', 'css', 'django', 'flask',
    'pandas', 'numpy', 'mongodb', 'mysql', 'aws', 'docker', 'jenkins',
    'adobe xd', 'figma', 'machine learning', 'deep learning'
]

# Skill extraction function
def extract_skills(text):
    text = text.lower()
    skills_found = [skill for skill in skill_keywords if skill in text]
    return list(set(skills_found))

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("resumes.csv")

# Train model
@st.cache_resource
def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data["resume_text"], data["label"], test_size=0.2, random_state=42
    )
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# Streamlit App UI
st.set_page_config(page_title="Resume Classifier", page_icon="🧠")
st.title("🧠 Resume Role Classifier + Skill Extractor")

data = load_data()
model, accuracy = train_model(data)

user_input = st.text_area("Paste resume summary or job experience:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter resume text.")
    else:
        prediction = model.predict([user_input])[0]
        skills = extract_skills(user_input)
        st.success(f"🎯 Predicted Job Role: **{prediction}**")
        st.info(f"🛠️ Extracted Skills: {', '.join(skills) if skills else 'None found'}")
        st.caption(f"Model Accuracy: {round(accuracy*100, 2)}%")

st.markdown("---")
st.caption("Project by Rithika Sri S | AI Resume Classifier")
