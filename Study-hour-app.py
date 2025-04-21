#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Loading the dataset
df = pd.read_csv("student-scores.csv")

# Creating average score per subject to find weak areas
subject_cols = ['math_score', 'history_score', 'physics_score',
                'chemistry_score', 'biology_score', 'english_score', 'geography_score']
df['Average Score'] = df[subject_cols].mean(axis=1)

print(df.head(5))


# In[2]:


#marking subjects where a student scores below average
def get_weak_subjects(row):
    weak_subjects = []
    for subject in subject_cols:
        if row[subject] < row['Average Score']:
            weak_subjects.append(subject.replace(' Score', ''))
    return weak_subjects

df['Weak Subjects'] = df.apply(get_weak_subjects, axis=1)


# In[3]:


import matplotlib.pyplot as plt
from collections import Counter

all_subjects = sum(df['Weak Subjects'], [])  
subject_counts = Counter(all_subjects)

plt.figure(figsize=(10,6))
plt.bar(subject_counts.keys(), subject_counts.values(), color='orchid')
plt.title("Most Common Weak Subjects Among Students")
plt.ylabel("Number of Students")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[4]:


def recommend_study_plan(row):
    weak = row['Weak Subjects']
    hours = row['weekly_self_study_hours']
    
    if hours < 5:
        suggestion = {subj: 2 for subj in weak}
    elif hours <= 10:
        suggestion = {subj: 1.5 for subj in weak}
    else:
        suggestion = {subj: 1 for subj in weak}
    
    return suggestion

df['Study Plan'] = df.apply(recommend_study_plan, axis=1)


# In[5]:


def show_study_plan(student_id):
    student = df[df['id'] == student_id].iloc[0]
    print(f"\n📘 Study Plan for {student['first_name']} {student['last_name']} (ID: {student_id})")
    print("Weak Subjects:", student['Weak Subjects'])
    print("Recommended Weekly Study Plan (in hrs):")
    for subject, hrs in student['Study Plan'].items():
        print(f"  - {subject}: {hrs} hrs")

#Viewing study plan for student with ID 12
show_study_plan(12)


# In[6]:


# Let's sum suggested hours from study plan
df['Total Suggested Hours'] = df['Study Plan'].apply(lambda x: sum(x.values()))

plt.hist(df['Total Suggested Hours'], bins=10, color='lavender', edgecolor='purple')
plt.title("Distribution of Total Suggested Study Hours")
plt.xlabel("Total Hours/Week")
plt.ylabel("Number of Students")
plt.grid(True)
plt.show()


# In[7]:


# Label ideal study hours 
def get_ideal_hours(avg_score):
    if avg_score >= 80:
        return 5
    elif avg_score >= 60:
        return 10
    elif avg_score >= 40:
        return 15
    else:
        return 20

df['Ideal Weekly Study Hours'] = df['Average Score'].apply(get_ideal_hours)


# In[8]:


# Convert boolean columns
df['part_time_job'] = df['part_time_job'].astype(int)
df['extracurricular_activities'] = df['extracurricular_activities'].astype(int)

X = df[['math_score', 'history_score', 'physics_score','chemistry_score', 'biology_score', 'english_score', 'geography_score',
        'part_time_job', 'absence_days', 'extracurricular_activities']]

y = df['Ideal Weekly Study Hours']


# In[9]:


#Training a regression model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f}")


# In[10]:


df['Predicted Study Hours'] = model.predict(X)

#Comparing actual ideal vs predicted
df[['Average Score', 'Ideal Weekly Study Hours', 'Predicted Study Hours']].head()


# In[11]:


import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Ideal Weekly Study Hours', y='Predicted Study Hours', data=df, hue='Average Score', palette='coolwarm')
plt.plot([0, 25], [0, 25], '--', color='gray')  # Line y = x
plt.title('Actual vs Predicted Study Hours')
plt.xlabel('Ideal Weekly Study Hours')
plt.ylabel('Predicted Study Hours')
plt.legend(title='Average Score')
plt.grid(True)
plt.show()


# In[12]:


import joblib

joblib.dump(model, "study_hour_predictor.pkl")


# In[13]:


from IPython.display import FileLink

FileLink(r'study_hour_predictor.pkl')


# In[14]:


app_code = """
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("study_hour_predictor.pkl")

st.set_page_config(page_title="📘 Study Hour Recommender", layout="centered")
st.title("📘 Study Hour Recommender App")

# Input form
st.subheader("📥 Enter Student Info")
math = st.slider("📐 Math Score", 0, 100, 70)
history = st.slider("📜 History Score", 0, 100, 70)
physics = st.slider("🔬 Physics Score", 0, 100, 70)
chemistry = st.slider("🧪 Chemistry Score", 0, 100, 70)
biology = st.slider("🧬 Biology Score", 0, 100, 70)
english = st.slider("📖 English Score", 0, 100, 70)
geography = st.slider("🌍 Geography Score", 0, 100, 70)
absence = st.number_input("📅 Absence Days", 0, 100, 3)
job = st.checkbox("👩‍💼 Has a Part-Time Job?")
activity = st.checkbox("🎨 Participates in Extracurricular Activities?")

# Prediction
if st.button("🔮 Recommend Study Hours"):
    input_data = pd.DataFrame([[math, history, physics, chemistry, biology, english, geography,
                                 int(job), absence, int(activity)]],
                               columns=['math_score', 'history_score', 'physics_score','chemistry_score', 'biology_score', 
                               'english_score', 'geography_score','part_time_job', 'absence_days', 
                               'extracurricular_activities'])

    predicted_hours = model.predict(input_data)[0]
    st.success(f"✨ Recommended Study Hours per Week: **{predicted_hours:.2f} hours**")

    # Visualization of score vs study hour importance
    score_data = {
        'Subjects': ['Math', 'History', 'Physics', 'Chemistry', 'Biology', 'English', 'Geography'],
        'Scores': [math, history, physics, chemistry, biology, english, geography]
    }
    score_df = pd.DataFrame(score_data)

    fig, ax = plt.subplots()
    sns.barplot(x='Scores', y='Subjects', data=score_df, palette='coolwarm', ax=ax)
    ax.set_title("📊 Subject Scores Overview")
    st.pyplot(fig)

    # Show pie chart for activities & job
    labels = ['Has Part-Time Job' if job else 'No Job', 'In Extracurriculars' if activity else 'No Activities']
    values = [1, 1]
    st.subheader("🧩 Lifestyle Factors")
    fig2, ax2 = plt.subplots()
    ax2.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    st.pyplot(fig2)
"""
with open("study_hour_app.py", "w") as f:
    f.write(app_code)


# In[ ]:


get_ipython().system('streamlit run study_hour_app.py')


# In[ ]:




