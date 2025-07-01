from flask import Flask, render_template, request
import pickle
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)

# Load the trained model, vectorizer, and encoder
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# Department to email mapping
department_emails = {
    "Hostel": "akshayhardie@gmail.com",
    "canteen": "kaizensarmy@gmail.com",
    "Academics": "vikramejjagiri2@gmail.com",
    "Library": "sandeeppathuri243@gmail.com",
    "Transport": "akshayhardie@gmail.com",
    "Others":"kaizensarmy@gmail.com"
    # Add more departments and email addresses as needed
}

# Function to classify the complaint
def classify_complaint(complaint):
    transformed = vectorizer.transform([complaint])
    prediction = model.predict(transformed)[0]
    return encoder.inverse_transform([prediction])[0]

# Function to send the complaint email to the respective department
def send_email(category, complaint):
    sender_email = "ejjagirivikram1327@gmail.com"
    sender_password = "osrdqvyauqzxseiu"  # App password
    recipient_email = department_emails.get(category, "default@gamil.com")  # fallback address

    subject = f"New Complaint - {category}"
    body = f"A new complaint has been submitted under the '{category}' category:\n\n{complaint}"

    # Email structure
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print(f"Email sent to {recipient_email}")
    except Exception as e:
        print("Error sending email:", e)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_complaint():
    complaint = request.form['complaint']
    if complaint:
        category = classify_complaint(complaint)
        send_email(category, complaint)
        return render_template('result.html', category=category, complaint=complaint)
    else:
        return render_template('index.html', error="Please enter a complaint.")

if __name__ == '__main__':
    app.run(debug=True)
