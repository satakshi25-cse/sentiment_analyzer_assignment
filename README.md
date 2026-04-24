# 🍽️ Restaurant Sentiment Analysis System  

## 📌 Overview  
This project is an advanced Sentiment Analysis System built using Natural Language Processing (NLP) and Machine Learning. It analyzes restaurant reviews and provides:  
- Overall sentiment (Positive, Negative, Neutral)  
- Restaurant-wise sentiment summary  
- Aspect-based sentiment (Food, Service, Ambience, Price)  
- Model performance evaluation  

Unlike basic sentiment classifiers, this project uses a hybrid approach combining rule-based and machine learning techniques for better accuracy and flexibility.

---

## 🚀 Key Features  

### 🔹 1. Text Preprocessing  
- Lowercasing  
- Punctuation removal  
- Stopword removal using NLTK  

### 🔹 2. Machine Learning Model  
- TF-IDF Vectorization (with n-grams)  
- Multinomial Naive Bayes classifier  
- Train-test split (80/20)  
- Cross-validation (CV = 5)  

### 🔹 3. Hybrid Sentiment Prediction  
- Rule-based detection (keywords like "good", "bad")  
- ML fallback for better generalization  

### 🔹 4. Restaurant-Level Analysis  
- Filters reviews by restaurant  
- Displays sentiment distribution  
- Generates pie chart visualization  
- Outputs overall sentiment  

### 🔹 5. Aspect-Based Sentiment Analysis  
Detects sentiment for:  
- 🍕 Food  
- 🧑‍🍳 Service  
- 🏡 Ambience  
- 💰 Price  

Uses contextual window-based analysis around keywords.

### 🔹 6. Aspect-wise Accuracy  
Evaluates model performance separately for each aspect.

---

## 🛠️ Tech Stack  

- Language: Python 🐍  
- Libraries:  
  - pandas  
  - nltk  
  - scikit-learn  
  - matplotlib  

---

## 📂 Project Structure  

bash sentiment-analysis/ │── review.csv              # Dataset │── sentiment_project.py    # Main script │── README.md               # Documentation 

---

## ⚙️ Installation  

bash git clone https://github.com/your-username/sentiment-analysis.git cd sentiment-analysis pip install pandas nltk scikit-learn matplotlib 

---

## ▶️ How to Run  

bash python sentiment_project.py 

---

## 🖥️ Menu Options  

The program provides an interactive CLI menu:

1. Analyze Review 2. Analyze Restaurant 3. Analyze Aspects 4. Aspect-wise Accuracy 5. Exit

---

## 📊 Example  

### Input:
The food was amazing but service was slow

### Output:
Aspect-wise Sentiment: Food: Positive Service: Negative

---

## 📈 Model Performance  

- Accuracy Score  
- Classification Report (Precision, Recall, F1-score)  
- Confusion Matrix  
- Cross-validation accuracy  

---

## 💡 Unique Highlights  

✔ Hybrid rule-based + ML approach  
✔ Aspect-based sentiment detection  
✔ Visualization using pie charts  
✔ Real-world use case (restaurant analytics)  
✔ Modular and extensible code  

---

## 🚧 Future Improvements  

- Use deep learning models (LSTM / Transformers)  
- Add GUI using Streamlit  
- Expand dataset for better accuracy  
- Deploy as a web application
