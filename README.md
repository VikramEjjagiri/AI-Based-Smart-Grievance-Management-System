
# 🧠 AI-Based Smart Grievance Management System

This project presents an **AI-powered Smart Grievance Management System** that classifies user complaints into predefined categories and automatically forwards them to relevant departments via email. It is built using Python, Flask, and Scikit-learn, and integrates machine learning models for real-time classification and automation.

## 🚀 Overview

The system uses supervised text classification with **TF-IDF vectorization** and applies **Logistic Regression** and **Random Forest** models to categorize complaints into labels such as:

- Academics  
- Canteen  
- Hostel  
- Library  
- Transport  
- Others  

An intuitive web interface built with Flask allows users to submit complaints in real time. Based on the classification output, the system sends the grievance to the appropriate department via email using SMTP.

## 🧪 Model Performance

The models were trained and validated using **70–30** and **80–20** train-test splits.

| Classifier         | Accuracy |
|--------------------|----------|
| Logistic Regression | 100%     |
| Random Forest       | 99%      |

All categories showed strong **precision**, **recall**, and **F1 scores**, with minimal misclassification observed in the **confusion matrix**.

## 🔧 Features

- 🔍 Intelligent text classification using ML models
- 📤 Real-time complaint submission via Flask web interface
- 📬 Automated email routing to relevant departments
- 📊 Confusion matrix and accuracy visualization
- 💾 Serialized model deployment (`model.pkl`, `vectorizer.pkl`, `encoder.pkl`)

## 📁 Project Structure

```
├── app.py                   # Flask backend
├── train_model.py          # ML training pipeline
├── vectorizer.pkl          # TF-IDF vectorizer
├── model.pkl               # Trained ML model
├── encoder.pkl             # Label encoder
├── templates/              # HTML templates (home and result pages)
├── static/                 # CSS or static files
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and instructions
├── USAGE_GUIDE.md          # (Optional) Usage and examples
└── deployment_instructions.md # (Optional) Deployment guide
```

## 🛠️ Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- SMTP (for email integration)
- HTML/CSS (Frontend)

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/smart-grievance-system.git
   cd smart-grievance-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:
   ```bash
   python app.py
   ```

5. Open your browser and go to `http://127.0.0.1:5000/`

## 📬 Email Setup

To enable automated email sending:
- Configure SMTP credentials inside `email_dispatcher.py` or within `app.py`
- Use a service like Gmail (with app passwords or less-secure app access)

## 🧪 Evaluation Tools

- Confusion Matrix Visualization
- Accuracy, Precision, Recall, F1-Score reporting
- Comparison of model performance (Random Forest vs Logistic Regression)

## 📌 Future Enhancements

- Admin dashboard for tracking complaints
- Feedback mechanism for students/staff
- Support for multilingual input
- Integration with institutional databases

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

**Keywords:**  
Supervised Text Classification, TF-IDF Vectorization, Ensemble Learning, Logistic Regression, Real-time NLP, Flask-based API, Email Automation, Scikit-learn Pipeline
