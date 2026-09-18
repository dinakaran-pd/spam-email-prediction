# spam-email-prediction

# 📧 Spam Email Prediction

An end-to-end **Machine Learning project** that classifies email or text messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) and machine learning techniques.

The project includes text preprocessing, TF-IDF feature extraction, model training, prediction, and an interactive **Streamlit web application**.

---

## 🚀 Project Overview

Spam emails are unwanted messages that can contain advertisements, scams, malicious links, or other irrelevant content.

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to automatically analyze the content of an email/message and predict whether it is:

* 🚨 **Spam**
* ✅ **Not Spam (Ham)**

The trained model can be accessed through an interactive Streamlit dashboard where users can enter an email/message and receive an instant prediction.

---

## ✨ Features

* 📩 Spam and ham email classification
* 🧹 Text preprocessing and cleaning
* 🔤 NLP-based text feature extraction
* 📊 TF-IDF vectorization
* 🤖 Machine Learning classification
* ⚡ Real-time prediction
* 🌐 Interactive Streamlit web interface
* 📈 Model evaluation
* 🖥️ User-friendly dashboard

---

## 🛠️ Technologies Used

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Programming language    |
| Pandas       | Data processing         |
| NumPy        | Numerical operations    |
| Scikit-learn | Machine Learning        |
| NLP          | Text processing         |
| TF-IDF       | Text feature extraction |
| Streamlit    | Web application         |
| Plotly       | Data visualization      |
| Git & GitHub | Version control         |

---

## 🧠 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Text Preprocessing
   ↓
Train / Test Split
   ↓
TF-IDF Vectorization
   ↓
Machine Learning Model
   ↓
Model Evaluation
   ↓
Prediction
   ↓
Streamlit Application
```

---

## 📂 Project Structure

```text
spam-email-prediction/
│
├── app.py
├── dataset.py
├── preprocessor.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── spam_email.csv
│
├── model/
│   └── trained_model.pkl
│
└── venv/
```

> **Note:** The actual file names and folders may vary depending on the project version.

---

## 🔍 Text Preprocessing

The email/message text is processed before it is provided to the machine learning model.

Typical preprocessing steps include:

1. Converting text to lowercase
2. Removing unnecessary characters
3. Removing unwanted spaces
4. Cleaning the input text
5. Converting text into numerical features

---

## 📊 TF-IDF Feature Extraction

Since machine learning models cannot directly understand raw text, the project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical feature vectors.

TF-IDF gives higher importance to words that are useful for distinguishing between different documents while reducing the importance of very common words.

```text
Email Text
    ↓
Text Cleaning
    ↓
TF-IDF Vectorizer
    ↓
Numerical Feature Vector
    ↓
ML Model
```

---

## 🤖 Prediction

The application accepts an email or message from the user.

Example:

```text
Congratulations! You have won a free prize.
Click here to claim your reward.
```

The model processes the text and returns:

```text
🚨 SPAM
```

Example of a normal message:

```text
Hi, can we meet tomorrow at 10 AM?
```

Prediction:

```text
✅ NOT SPAM
```

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit interface.

Users can:

* Enter an email/message
* Submit the text for analysis
* View the predicted category
* Understand whether the message is potentially spam

### Run the application

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Then start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/dinakaran-pd/spam-email-prediction.git
```

### 2. Navigate to the project

```bash
cd spam-email-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```bash
venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

---

## 📈 Model Evaluation

The machine learning model can be evaluated using commonly used classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

These metrics help evaluate how effectively the model distinguishes spam messages from legitimate messages.

---

## 🎯 Project Objectives

* Build a practical NLP-based machine learning application
* Understand text preprocessing
* Implement TF-IDF feature extraction
* Train and evaluate a classification model
* Deploy the model through Streamlit
* Develop an end-to-end machine learning workflow

---

## 🔮 Future Improvements

Some possible improvements include:

* 🔹 Compare multiple classification algorithms
* 🔹 Improve NLP preprocessing
* 🔹 Add probability/confidence scores
* 🔹 Add email file upload support
* 🔹 Add visualization of important words
* 🔹 Improve the Streamlit dashboard
* 🔹 Deploy the application online
* 🔹 Add deep learning-based text classification

---

## 👨‍💻 Author

**P. Dinakaran**

B.Tech – Artificial Intelligence & Data Science

Interested in:

* Artificial Intelligence
* Machine Learning
* Data Science
* Natural Language Processing
* Deep Learning

---

## ⭐ If you find this project useful

Give the repository a ⭐ on GitHub and feel free to explore the project!
