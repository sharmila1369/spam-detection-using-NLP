# 📧 Spam Detection using NLP (Machine Learning Web App)

---

## 🔷 Project Overview

This project is an end-to-end **Spam Message Classification System** built using **Machine Learning and Natural Language Processing (NLP)** with a **Flask-based web interface**.  
The system classifies a given text message as **“Spam” or “Not Spam”** based on a trained ML model.

The project covers the complete ML pipeline including:
- Data preprocessing  
- Feature engineering  
- Model training & evaluation  
- Model serialization  
- Web application development using Flask  

---

## 🔷 Problem Statement

With the rapid increase in digital communication, spam messages have become a major issue.  
This project aims to automatically detect and filter spam messages using machine learning techniques to enhance security and user experience.

---

## 🔷 Tech Stack

**Programming Language:**
- Python  

**Libraries & Frameworks:**
- Flask  
- Scikit-learn  
- Pandas  
- NumPy  
- NLTK / Regex (for text preprocessing)

**Machine Learning Model:**
- Multinomial Naive Bayes  

**Text Representation:**
- TF-IDF Vectorization  

**Frontend:**
- HTML, CSS  

---

## 🔷 Dataset

- Used a labeled dataset containing SMS messages categorized as:
  - `ham` → Not Spam  
  - `spam` → Spam  
- Dataset underwent cleaning and preprocessing before model training.

---

## 🔷 Data Preprocessing Steps

1. Convert text to lowercase  
2. Remove special characters and punctuation  
3. Remove stopwords  
4. Tokenization  
5. Convert text into numerical format using **TF-IDF Vectorizer**

---

## 🔷 Model Training & Evaluation

- **Algorithm Used:** Multinomial Naive Bayes  
- **Train-Test Split:** 80% training, 20% testing  

**Evaluation Metrics:**
- Accuracy Score  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 🔷 Model Deployment

- The trained model was serialized using `pickle`  
- Integrated with a Flask web application  
- Users can input a message via a web interface and receive real-time predictions  

---

## 🔷 System Architecture
# 📧 Spam Detection using NLP (Machine Learning Web App)

---

## 🔷 Project Overview

This project is an end-to-end **Spam Message Classification System** built using **Machine Learning and Natural Language Processing (NLP)** with a **Flask-based web interface**.  
The system classifies a given text message as **“Spam” or “Not Spam”** based on a trained ML model.

The project covers the complete ML pipeline including:
- Data preprocessing  
- Feature engineering  
- Model training & evaluation  
- Model serialization  
- Web application development using Flask  

---

## 🔷 Problem Statement

With the rapid increase in digital communication, spam messages have become a major issue.  
This project aims to automatically detect and filter spam messages using machine learning techniques to enhance security and user experience.

---

## 🔷 Tech Stack

**Programming Language:**
- Python  

**Libraries & Frameworks:**
- Flask  
- Scikit-learn  
- Pandas  
- NumPy  
- NLTK / Regex (for text preprocessing)

**Machine Learning Model:**
- Multinomial Naive Bayes  

**Text Representation:**
- TF-IDF Vectorization  

**Frontend:**
- HTML, CSS  

---

## 🔷 Dataset

- Used a labeled dataset containing SMS messages categorized as:
  - `ham` → Not Spam  
  - `spam` → Spam  
- Dataset underwent cleaning and preprocessing before model training.

---

## 🔷 Data Preprocessing Steps

1. Convert text to lowercase  
2. Remove special characters and punctuation  
3. Remove stopwords  
4. Tokenization  
5. Convert text into numerical format using **TF-IDF Vectorizer**

---

## 🔷 Model Training & Evaluation

- **Algorithm Used:** Multinomial Naive Bayes  
- **Train-Test Split:** 80% training, 20% testing  

**Evaluation Metrics:**
- Accuracy Score  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 🔷 Model Deployment

- The trained model was serialized using `pickle`  
- Integrated with a Flask web application  
- Users can input a message via a web interface and receive real-time predictions  

---

## 🔷 System Architecture

# 📧 Spam Detection using NLP (Machine Learning Web App)

---

## 🔷 Project Overview

This project is an end-to-end **Spam Message Classification System** built using **Machine Learning and Natural Language Processing (NLP)** with a **Flask-based web interface**.  
The system classifies a given text message as **“Spam” or “Not Spam”** based on a trained ML model.

The project covers the complete ML pipeline including:
- Data preprocessing  
- Feature engineering  
- Model training & evaluation  
- Model serialization  
- Web application development using Flask  

---

## 🔷 Problem Statement

With the rapid increase in digital communication, spam messages have become a major issue.  
This project aims to automatically detect and filter spam messages using machine learning techniques to enhance security and user experience.

---

## 🔷 Tech Stack

**Programming Language:**
- Python  

**Libraries & Frameworks:**
- Flask  
- Scikit-learn  
- Pandas  
- NumPy  
- NLTK / Regex (for text preprocessing)

**Machine Learning Model:**
- Multinomial Naive Bayes  

**Text Representation:**
- TF-IDF Vectorization  

**Frontend:**
- HTML, CSS  

---

## 🔷 Dataset

- Used a labeled dataset containing SMS messages categorized as:
  - `ham` → Not Spam  
  - `spam` → Spam  
- Dataset underwent cleaning and preprocessing before model training.

---

## 🔷 Data Preprocessing Steps

1. Convert text to lowercase  
2. Remove special characters and punctuation  
3. Remove stopwords  
4. Tokenization  
5. Convert text into numerical format using **TF-IDF Vectorizer**

---

## 🔷 Model Training & Evaluation

- **Algorithm Used:** Multinomial Naive Bayes  
- **Train-Test Split:** 80% training, 20% testing  

**Evaluation Metrics:**
- Accuracy Score  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 🔷 Model Deployment

- The trained model was serialized using `pickle`  
- Integrated with a Flask web application  
- Users can input a message via a web interface and receive real-time predictions  

---

## 🔷 System Architecture
User Input → Text Preprocessing → TF-IDF Vectorization → Trained ML Model → Prediction → Flask UI Output

---

## 🔷 Project Structure
spam-detection-using-NLP/
│-- app.py
│-- model.pkl
│-- vectorizer.pkl
│-- requirements.txt
│-- templates/
│ └── index.html
│-- static/
│ └── style.css
│-- README.md


---

## 🔷 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sharmila1369/spam-detection-using-NLP.git
cd spam-detection-using-NLP

2️⃣ Create Virtual Environment (Optional)
python -m venv venv
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
python app.py


Open browser:

http://127.0.0.1:5000/

# 📊 Results

- Successfully classifies messages as **Spam or Not Spam**  
- Provides fast and reliable predictions  
- User-friendly interface  

---

# 🚀 Future Enhancements

- Train with a larger dataset  
- Implement Deep Learning models (LSTM, BERT)  
- Add multilingual spam detection  
- Deploy on AWS / Heroku / Render  
- Add API support  

---

# ✍️ Author

**Sharmilambika Venna**  
B.Tech – CSE (AI & Data Science)  

GitHub: https://github.com/sharmila1369  
LinkedIn: https://www.linkedin.com/in/sharmilambika-venna/

---


