import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the model
model = joblib.load("study_hour_predictor.pkl")

st.title("📘 Study Hour Recommender")
st.subheader("Predict ideal self-study hours based on student performance")

# Collect input
math = st.slider("Math Score", 0, 100, 70)
english = st.slider("English Score", 0, 100, 70)
biology = st.slider("Biology Score", 0, 100, 70)
chemistry = st.slider("Chemistry Score", 0, 100, 70)
physics = st.slider("Physics Score", 0, 100, 70)
geography = st.slider("Geography Score", 0, 100, 70)
history = st.slider("History Score", 0, 100, 70)

absence_days = st.number_input("Days Absent", 0, 100, 5)
part_time = st.selectbox("Has a part-time job?", ["Yes", "No"])
extracurricular = st.selectbox("In extracurricular activities?", ["Yes", "No"])

# Prepare input data
input_data = pd.DataFrame([{
    'Math Score': math,
    'English Score': english,
    'Biology Score': biology,
    'Chemistry Score': chemistry,
    'Physics Score': physics,
    'Geography Score': geography,
    'History Score': history,
    'Absence Days': absence_days,
    'Part-Time Job': 1 if part_time == "Yes" else 0,
    'Extracurricular Activities': 1 if extracurricular == "Yes" else 0
}])

# Predict
if st.button("Predict Study Hours"):
    predicted_hours = model.predict(input_data)[0]
    st.success(f"Recommended Weekly Study Hours: {predicted_hours:.2f} 📚")

    # 🎨 Visualize
    st.subheader("📊 Score Overview")
    scores = input_data.iloc[0][['Math Score', 'English Score', 'Biology Score', 
                                 'Chemistry Score', 'Physics Score', 'Geography Score', 
                                 'History Score']]
    fig, ax = plt.subplots()
    ax.bar(scores.index, scores.values, color='lavender')
    ax.set_ylabel('Score')
    ax.set_title('Subject Scores')
    plt.xticks(rotation=45)
    st.pyplot(fig)
