import pandas as pd
import string
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("review.csv")

# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

data['Cleaned'] = data['Review'].apply(preprocess)

# -----------------------------
# TRAIN MODEL
# -----------------------------
X = data['Cleaned']
y = data['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# MODEL EVALUATION
# -----------------------------
y_pred = model.predict(X_test_vec)

print("\nMODEL PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Cross-validation
scores = cross_val_score(model, vectorizer.transform(X), y, cv=5)
print("\nCross-validation Accuracy:", scores.mean())

# -----------------------------
# REVIEW SENTIMENT (HYBRID)
# -----------------------------
def analyze_review(review):
    review_lower = review.lower().strip()

    # Rule-based
    negative_words = ["bad", "worst", "horrible", "terrible", "poor", "not good"]
    positive_words = ["good", "amazing", "excellent", "tasty", "great", "love"]
    neutral_words = ["okay", "average", "fine", "decent", "not bad"]

    for w in negative_words:
        if w in review_lower:
            return "Negative"

    for w in positive_words:
        if w in review_lower:
            return "Positive"

    for w in neutral_words:
        if w in review_lower:
            return "Neutral"

    # ML fallback
    cleaned = preprocess(review)
    vector = vectorizer.transform([cleaned])
    return model.predict(vector)[0]

# -----------------------------
# RESTAURANT ANALYSIS
# -----------------------------
def analyze_restaurant(name):
    filtered = data[data['Restaurant'].str.lower() == name.lower()]

    if filtered.empty:
        print("No data found")
        return

    sentiments = filtered['Sentiment'].value_counts()

    print("\nSentiment Summary:")
    print(sentiments)

    # Graph
    sentiments.plot.pie(autopct='%1.1f%%')
    plt.title(f"Sentiment for {name}")
    plt.ylabel("")
    plt.show()

    print("\n Overall Sentiment:", sentiments.idxmax())

# -----------------------------
# ASPECT-BASED ANALYSIS
# -----------------------------
def analyze_aspects(review):
    review_lower = review.lower()

    aspects = {
        "food": ["food", "taste", "pizza", "meal"],
        "service": ["service", "staff", "waiter"],
        "ambience": ["ambience", "environment", "place", "atmosphere"],
        "money": ["price", "cost", "expensive", "cheap", "worth","higher","high"]
    }

    results = {}

    for aspect, keywords in aspects.items():
        for word in keywords:
            if word in review_lower:
                words=review_lower.split()
                idx=words.index(word)
                start=max(0,idx-3)
                end=min(len(words),idx+4)
                context=" ".join(words[start:end])
                sentiment=analyze_review(context)
                results[aspect]=sentiment
                break
    return results

# -----------------------------
# ASPECT-WISE ACCURACY
# -----------------------------
def aspect_accuracy():
    print("\nAspect-wise Accuracy:")

    for aspect in data['Aspect'].unique():
        subset = data[data['Aspect'] == aspect]

        X_subset = subset['Cleaned']
        y_true = subset['Sentiment']

        X_vec = vectorizer.transform(X_subset)
        y_pred = model.predict(X_vec)

        acc = accuracy_score(y_true, y_pred)

        print(f"{aspect}: {acc:.2f}")

# -----------------------------
# MENU
# -----------------------------
while True:
    print("\n===== MENU =====")
    print("1. Analyze Review")
    print("2. Analyze Restaurant")
    print("3. Analyze Aspects")
    print("4. Aspect-wise Accuracy")
    print("5. Exit")

    choice = input("Enter choice: ")

    if choice == '1':
        review = input("Enter review: ")
        print("Sentiment:", analyze_review(review))

    elif choice == '2':
        name = input("Enter restaurant name: ")
        analyze_restaurant(name)

    elif choice == '3':
        review = input("Enter review: ")
        result = analyze_aspects(review)

        print("\nAspect-wise Sentiment:")
        for k, v in result.items():
            print(f"{k}: {v}")

    elif choice == '4':
        aspect_accuracy()

    elif choice == '5':
        break