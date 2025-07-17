# 🧠 Depression Risk Predictor and Care

A Streamlit-based web application that predicts the risk of depression based on lifestyle habits, social media behavior, and emotional wellbeing — with care tips, mood tracking, SHAP explainability, and admin dashboard.

---

## 🚀 Features

- 🌐 **Web UI**: Built with Streamlit for real-time input and predictions
- 🤖 **ML Model**: Trained using Random Forest and SHAP explainability
- 📊 **Mood Tracker**: Daily check-ins and progress logging
- 📥 **Admin Dashboard**: Logs, mood trends, risk summaries
- 📧 **Email Reminder Generator**: Auto-generate motivational messages
- 🔐 **Admin-only Controls**: Secure mood log download and analytics
- 📈 **SHAP Visualizations** *(admin only)*

---

## 🧪 How It Works

1. User enters:
   - Age, Gender, Occupation
   - Sleep hours, Physical activity, Social media use
   - Sentiment from latest social post
2. Model predicts risk: ✅ Not At Risk or ⚠️ At Risk
3. Care suggestions + logging happens automatically
4. Admins can view overall logs and trends

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- `pip install -r requirements.txt`

### Run the App Locally

```bash
streamlit run app.py

🔐 Admin Access
To access analytics and logs, enter this admin password when prompted:
1234

🧠 SHAP Explainability
SHAP visualizations help explain the model’s prediction

Visible only to admin users for privacy


📄 License
This project is licensed under the MIT License.
See the LICENSE file for details.

👤 Author
Rohit U
Built as part of a complete ML + Streamlit deployment journey.


# Example Docker Usage:
docker build -t depression-app .
docker run -p 8501:8501 depression-app
