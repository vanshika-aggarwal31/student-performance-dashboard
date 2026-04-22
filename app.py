import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

st.set_page_config(page_title="Student Performance Dashboard", layout="wide", page_icon="🎓")

st.markdown("""
<style>
.main { background-color: #f0f2f6; }
.metric-card { background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/StudentsPerformance.csv")
    df.columns = df.columns.str.strip()
    df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
    return df

df = load_data()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
st.sidebar.title("🎓 Student Dashboard")
page = st.sidebar.radio("Navigate", ["📊 Overview", "📈 Analysis", "🔮 Predict Score"])

# =================== PAGE 1: OVERVIEW ===================
if page == "📊 Overview":
    st.title("📊 Student Performance Overview")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Avg Math Score", f"{df['math score'].mean():.1f}")
    with col3:
        st.metric("Avg Reading Score", f"{df['reading score'].mean():.1f}")
    with col4:
        st.metric("Avg Writing Score", f"{df['writing score'].mean():.1f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📚 Subject-wise Average Scores")
        fig, ax = plt.subplots(figsize=(6,4))
        subjects = ['math score', 'reading score', 'writing score']
        scores = [df[s].mean() for s in subjects]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.bar(['Math', 'Reading', 'Writing'], scores, color=colors, edgecolor='white', linewidth=1.5)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{score:.1f}', ha='center', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_ylabel("Average Score")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    with col2:
        st.subheader("👥 Gender Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        gender_counts = df['gender'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4']
        wedges, texts, autotexts = ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        for text in autotexts:
            text.set_fontweight('bold')
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🏆 Score Distribution")
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for ax, subject, color in zip(axes, subjects, colors):
        ax.hist(df[subject], bins=20, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(df[subject].mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: {df[subject].mean():.1f}')
        ax.set_title(subject.replace(' score', '').title())
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

# =================== PAGE 2: ANALYSIS ===================
elif page == "📈 Analysis":
    st.title("📈 Deep Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👥 Gender vs Scores")
        fig, ax = plt.subplots(figsize=(7,4))
        gender_scores = df.groupby('gender')[['math score', 'reading score', 'writing score']].mean()
        gender_scores.plot(kind='bar', ax=ax, color=['#FF6B6B','#4ECDC4','#45B7D1'], edgecolor='white')
        ax.set_xticklabels(['Female', 'Male'], rotation=0)
        ax.set_ylim(0, 100)
        ax.legend(['Math', 'Reading', 'Writing'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    with col2:
        st.subheader("🍽️ Lunch Type vs Performance")
        fig, ax = plt.subplots(figsize=(7,4))
        lunch_scores = df.groupby('lunch')[['math score', 'reading score', 'writing score']].mean()
        lunch_scores.plot(kind='bar', ax=ax, color=['#FF6B6B','#4ECDC4','#45B7D1'], edgecolor='white')
        ax.set_xticklabels(lunch_scores.index, rotation=0)
        ax.set_ylim(0, 100)
        ax.legend(['Math', 'Reading', 'Writing'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🎓 Parent Education Impact")
    fig, ax = plt.subplots(figsize=(12,5))
    edu_scores = df.groupby('parental level of education')['average_score'].mean().sort_values(ascending=True)
    bars = ax.barh(edu_scores.index, edu_scores.values, color='#4ECDC4', edgecolor='white')
    for bar, score in zip(bars, edu_scores.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, f'{score:.1f}', va='center', fontweight='bold')
    ax.set_xlabel("Average Score")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📝 Test Preparation Effect")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        prep_scores = df.groupby('test preparation course')['average_score'].mean()
        colors = ['#FF6B6B', '#4ECDC4']
        bars = ax.bar(prep_scores.index, prep_scores.values, color=colors, edgecolor='white')
        for bar, score in zip(bars, prep_scores.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{score:.1f}', ha='center', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_ylabel("Average Score")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
    with col2:
        completed = df[df['test preparation course']=='completed']['average_score'].mean()
        none = df[df['test preparation course']=='none']['average_score'].mean()
        diff = completed - none
        st.markdown("### 💡 Key Insight")
        st.info(f"""
        Students who completed test preparation scored **{diff:.1f} points higher** on average!
        
        - ✅ Completed prep: **{completed:.1f}/100**
        - ❌ No prep: **{none:.1f}/100**
        """)

# =================== PAGE 3: PREDICTION ===================
elif page == "🔮 Predict Score":
    st.title("🔮 Predict Student Score")
    st.markdown("---")

    # Train model
    le_dict = {}
    df_model = df.copy()
    cat_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le

    X = df_model[cat_cols]
    y = df_model['average_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))

    st.success(f"🎯 Model Accuracy (R² Score): **{r2*100:.1f}%**")
    st.markdown("---")

    st.subheader("Enter Student Details:")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("👤 Gender", df['gender'].unique())
        race = st.selectbox("🌍 Race/Ethnicity", df['race/ethnicity'].unique())
        parent_edu = st.selectbox("🎓 Parent Education", df['parental level of education'].unique())

    with col2:
        lunch = st.selectbox("🍽️ Lunch Type", df['lunch'].unique())
        test_prep = st.selectbox("📝 Test Preparation", df['test preparation course'].unique())

    if st.button("🎯 Predict Score", use_container_width=True):
        input_dict = {
            'gender': [gender],
            'race/ethnicity': [race],
            'parental level of education': [parent_edu],
            'lunch': [lunch],
            'test preparation course': [test_prep]
        }
        input_df = pd.DataFrame(input_dict)
        for col in cat_cols:
            input_df[col] = le_dict[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]
        prediction = max(0, min(100, prediction))

        st.markdown("---")
        st.subheader("📊 Prediction Result:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Average Score", f"{prediction:.1f}/100")

        if prediction >= 70:
            st.success(f"🌟 Great! Predicted score is {prediction:.1f}/100 — Excellent Performance!")
        elif prediction >= 50:
            st.warning(f"📚 Predicted score is {prediction:.1f}/100 — Average Performance")
        else:
            st.error(f"⚠️ Predicted score is {prediction:.1f}/100 — Needs Improvement")