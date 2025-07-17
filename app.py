import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
import shap
import matplotlib.pyplot as plt
import os
import streamlit.components.v1 as components

# Load model, scaler, and expected feature columns
model = joblib.load('models/final_model.pkl')
scaler = joblib.load('models/final_scaler.pkl')
expected_columns = joblib.load('models/final_features_columns.pkl')

# App layout
st.set_page_config(page_title="Depression Risk Predictor", layout="centered")
st.title("🧠 Depression Risk Predictor")

# --- USER INPUT ---
age = st.slider("Age", 12, 80, 20)
gender = st.selectbox("Gender", ["Male", "Female"])
occupation = st.selectbox("Occupation", ["Student", "Working", "Unemployed", "Other"])
sleep_hours = st.slider("Average Sleep Hours", 0, 12, 6)
activity_level = st.slider("Physical Activity Level (0–10)", 0, 10, 5)
social_time = st.slider("Daily Social Media Usage (hrs)", 0, 12, 3)
post_text = st.text_area("Recent Social Media Post")

def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

sentiment_score = get_sentiment(post_text)

# Prepare input
data = {col: 0 for col in expected_columns}
data['age'] = age
data['sleep_hours'] = sleep_hours
data['activity_level'] = activity_level
data['social_media_time'] = social_time
data['sentiment_score'] = sentiment_score

# Encode gender
if 'gender_Male' in expected_columns:
    data['gender_Male'] = 1 if gender == 'Male' else 0
if 'gender_Female' in expected_columns:
    data['gender_Female'] = 1 if gender == 'Female' else 0

# Encode occupation
occ_col = f"occupation_{occupation}"
if occ_col in expected_columns:
    data[occ_col] = 1

input_df = pd.DataFrame([data])[expected_columns]
input_scaled = scaler.transform(input_df)

# --- PREDICTION ---
if st.button("🔮 Predict My Risk", key="predict_button"):
    prediction = model.predict(input_scaled)[0]
    label = "⚠️ At Risk" if prediction == 1 else "✅ Not At Risk"

    st.subheader("Result:")
    st.success(f"Your predicted risk level is: **{label}**")

    # Log prediction
    log_data = input_df.copy()
    log_data['prediction'] = label
    log_data['timestamp'] = pd.Timestamp.now()

    log_file = "data/user_logs.csv"
    os.makedirs("data", exist_ok=True)
    try:
        pd.read_csv(log_file)
        log_data.to_csv(log_file, mode='a', header=False, index=False)
    except FileNotFoundError:
        log_data.to_csv(log_file, mode='w', header=True, index=False)

    # --- Care Recommendations ---
    st.markdown("---")
    st.subheader("💡 Care Recommendations")
    if prediction == 1:
        st.warning("You seem to be at risk. Here are some helpful suggestions:")
        st.markdown("""
        - 💤 Maintain regular sleep (7–9 hours).
        - 🏃 Get 20+ minutes of daily activity.
        - 📵 Limit social media, especially before bed.
        - 💬 Talk to a friend, counselor, or family.
        - 🧘 Try mindfulness or journaling.
        """)
    else:
        st.success("You're doing great! Keep it up.")
        st.markdown("""
        - ✔️ Continue healthy sleep & activity.
        - 🤝 Support friends in need.
        - 📓 Track your mood and feelings.
        """)

# --- Mood Check-In ---
st.header("📅 Daily Mood Check-In")
mood = st.radio("How are you feeling today?", ["😊 Good", "😐 Okay", "😢 Not Good"], key="daily_mood")
if st.button("Submit Mood", key="submit_mood_button"):
    mood_data = {
        "timestamp": pd.Timestamp.now(),
        "mood": mood,
        "age": age,
        "gender": gender,
        "sleep_hours": sleep_hours,
        "activity_level": activity_level,
        "social_media_time": social_time,
        "sentiment_score": sentiment_score,
        "post_text": post_text
    }
    mood_df = pd.DataFrame([mood_data])
    try:
        mood_df.to_csv("data/mood_logs.csv", mode="a", index=False, header=not os.path.exists("data/mood_logs.csv"))
        st.success("✅ Mood submitted successfully!")
    except Exception as e:
        st.error(f"⚠️ Failed to save mood: {e}")

# --- Mood Trends ---
st.header("📊 Mood Trend Over Time")
if st.button("📖 Show Mood Logs", key="show_mood_logs"):
    try:
        mood_logs = pd.read_csv("data/mood_logs.csv", parse_dates=["timestamp"])
        mood_map = {"😊 Good": 2, "😐 Okay": 1, "😢 Not Good": 0}
        mood_logs['mood_score'] = mood_logs['mood'].map(mood_map)
        st.line_chart(mood_logs.set_index("timestamp")["mood_score"])
    except FileNotFoundError:
        st.warning("No mood logs found yet.")

# --- Admin Section ---
st.markdown("---")
st.header("🔐 Admin Dashboard")
admin_access = st.text_input("Enter admin passcode", type="password")

if admin_access == "1234":
    st.success("Welcome Admin!")

    try:
        logs = pd.read_csv("data/user_logs.csv")
        logs['timestamp'] = pd.to_datetime(logs['timestamp'])
        st.subheader("📂 All User Logs")
        st.dataframe(logs.tail(100))

        st.subheader("📊 Overall Risk Prediction Count")
        st.bar_chart(logs['prediction'].value_counts())

    except FileNotFoundError:
        st.warning("No user logs found.")

    try:
        mood_logs = pd.read_csv("data/mood_logs.csv")
        mood_logs['timestamp'] = pd.to_datetime(mood_logs['timestamp'])

        st.subheader("📈 Mood Trends Over Time")
        mood_summary = mood_logs.groupby(mood_logs['timestamp'].dt.date)['mood'].value_counts().unstack().fillna(0)
        st.line_chart(mood_summary)

    except FileNotFoundError:
        st.warning("No mood logs found.")

    # --- SHAP Explainability ---
    st.subheader("📌 SHAP Explainability")
    explainer = shap.Explainer(model, input_df)
    shap_values = explainer(input_df)

    shap.summary_plot(shap_values, input_df, show=False)
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(bbox_inches='tight')

    # --- Export Logs ---
    st.header("📤 Export My Logs")
    if st.button("📁 Download My Logs", key="download_logs"):
        try:
            logs = pd.read_csv("data/user_logs.csv")
            st.download_button("📥 Click to Download", data=logs.to_csv(index=False), file_name="depression_logs.csv", mime="text/csv")
        except FileNotFoundError:
            st.warning("No log data available to export.")

# --- EMAIL REMINDER TEXT GENERATOR ---
st.header("📧 Generate Email Reminder")
user_name = st.text_input("Enter your name:")
email_purpose = st.selectbox("Purpose of Email", ["Daily Reminder", "Progress Summary", "Check-In Message"])
custom_msg = st.text_area("Custom Message (Optional)", "")

if st.button("✉️ Generate Email Text", key="email_text"):
    body = f"Dear {user_name},\n\n"

    if email_purpose == "Daily Reminder":
        body += "This is your gentle reminder to check in with your mood and wellbeing today. 😊"
    elif email_purpose == "Progress Summary":
        try:
            logs = pd.read_csv("data/user_logs.csv")
            total = len(logs)
            at_risk = (logs["prediction"] == "⚠️ At Risk").sum()
            body += f"So far, you've logged {total} entries.\n"
            body += f"Of those, {at_risk} showed signs of risk.\n\nKeep tracking your wellness!"
        except:
            body += "We couldn't find your logs yet. Please make your first prediction today!"
    elif email_purpose == "Check-In Message":
        body += "Hope you're doing well! Remember to care for yourself today — even small steps count."

    if custom_msg:
        body += "\n\nCustom message:\n" + custom_msg

    body += "\n\n— Your Depression Risk App 💙"
    st.code(body, language="text")
