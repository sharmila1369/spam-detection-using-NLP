📧 Spam Classifier – Machine Learning Web Application
🔹 Project Overview

This project is an end-to-end Spam Message Classification System built using Machine Learning and Natural Language Processing (NLP) with a Flask-based web interface. The system classifies a given text message as “Spam” or “Not Spam” based on a trained ML model.

The project covers the complete ML pipeline including:

Data preprocessing

Feature engineering

Model training & evaluation

Model serialization

Web application development using Flask

🔹 Problem Statement

With the rapid increase in digital communication, spam messages have become a major issue. This project aims to automatically detect and filter spam messages using machine learning techniques to enhance security and user experience.

🔹 Tech Stack

Programming Language: Python

Libraries & Frameworks:

Flask

Scikit-learn

Pandas

NumPy

NLTK / Regex (for text preprocessing)

Machine Learning Model:

Multinomial Naive Bayes (primary model)

Text Representation:

TF-IDF Vectorization

Frontend:

HTML, CSS

🔹 Dataset

Used a labeled dataset containing SMS messages categorized as:

ham → Not Spam

spam → Spam

Dataset underwent cleaning and preprocessing before model training.

🔹 Data Preprocessing Steps

The following NLP preprocessing techniques were applied:

Convert text to lowercase

Remove special characters and punctuation

Remove stopwords

Tokenization

Convert text into numerical format using TF-IDF Vectorizer

🔹 Model Training & Evaluation

Algorithm Used: Multinomial Naive Bayes

Train-Test Split: 80% training, 20% testing

Evaluation Metrics:

Accuracy Score

Precision

Recall

F1-score

Confusion Matrix

The model achieved high accuracy and performed well on unseen test data.

🔹 Model Deployment

The trained model was serialized using pickle

Integrated with a Flask web application

Users can input a message via a web interface and receive real-time predictions

🔹 System Architecture
User Input → Text Preprocessing → TF-IDF Vectorization → Trained ML Model → Prediction → Flask UI Output
Project Structure:
Spam-Classifier/
│-- app.py
│-- model.pkl
│-- vectorizer.pkl
│-- requirements.txt
│-- templates/
│   └── index.html
│-- static/
│   └── style.css
│-- README.md

How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier

2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
python app.py


Then open your browser and visit:

http://127.0.0.1:5000/

🔹 Results

Successfully classifies messages as Spam or Not Spam

Provides fast and reliable predictions

User-friendly interface

🔹 Future Enhancements

Train with a larger dataset

Implement Deep Learning models (LSTM, BERT)

Add multilingual spam detection

Deploy on AWS / Heroku / Render

Add API support

🔹 Author

Your Name
Venna Sharmilambika
B.Tech – CSE (AI & Data Science)

LinkedIn:www.linkedin.com/in/sharmilambika-venna13



