import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# Page config
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("🎓 Student Performance Prediction Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/StudentsPerformance.csv")
    return df

df = load_data()

# Preprocessing
le = LabelEncoder()
df_model = df.copy()
for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le.fit_transform(df_model[col])

df_model['average_score'] = (df_model['math score'] + 
                              df_model['reading score'] + 
                              df_model['writing score']) / 3

# Features and target
X = df_model[['math score', 'reading score']]
y = df_model['writing score']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Sidebar
st.sidebar.header("📊 Filter Data")
gender = st.sidebar.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())
lunch = st.sidebar.multiselect("Lunch Type", df['lunch'].unique(), default=df['lunch'].unique())
filtered_df = df[(df['gender'].isin(gender)) & (df['lunch'].isin(lunch))]

# Metrics
st.subheader("📈 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score (Accuracy)", f"{r2*100:.2f}%")
col2.metric("Mean Absolute Error", f"{mae:.2f}")
col3.metric("Total Students", len(filtered_df))

st.markdown("---")

# Charts
st.subheader("📊 Score Distribution")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    filtered_df[['math score', 'reading score', 'writing score']].mean().plot(kind='bar', ax=ax, color=['#FF6B6B','#4ECDC4','#45B7D1'])
    ax.set_title("Average Scores by Subject")
    ax.set_ylabel("Score")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['math score'], kde=True, ax=ax, color='#FF6B6B')
    ax.set_title("Math Score Distribution")
    st.pyplot(fig)

st.markdown("---")

# Gender comparison
st.subheader("👥 Gender-wise Performance")
fig, ax = plt.subplots()
filtered_df.groupby('gender')[['math score', 'reading score', 'writing score']].mean().plot(kind='bar', ax=ax)
ax.set_title("Gender vs Average Scores")
ax.set_ylabel("Score")
plt.xticks(rotation=0)
st.pyplot(fig)

st.markdown("---")

# Prediction section
st.subheader("🔮 Predict Student Performance")
col1, col2 = st.columns(2)

with col1:
    gender_input = st.selectbox("Gender", ["male", "female"])
    race_input = st.selectbox("Race/Ethnicity", df['race/ethnicity'].unique())
    parent_edu = st.selectbox("Parent Education", df['parental level of education'].unique())

with col2:
    lunch_input = st.selectbox("Lunch", ["standard", "free/reduced"])
    test_prep = st.selectbox("Test Preparation", ["none", "completed"])

if st.button("🎯 Predict Score"):
    input_data = pd.DataFrame([[gender_input, race_input, parent_edu, lunch_input, test_prep]],
                               columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])
    for col in input_data.columns:
        input_data[col] = le.fit_transform(input_data[col])
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Predicted Average Score: {prediction:.2f} / 100")

st.markdown("---")
st.caption("Built with Python, Scikit-learn & Streamlit")