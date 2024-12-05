import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import re
import numpy as np

#datafile = "reduced_data.json"
datafile = "yelp_academic_dataset_review.json"
important_fields = ["stars", "useful", "funny", "cool", "text"]
unimportant_fields = ["review_id", "user_id", "business_id", "date"]

#process a single line of the json data and return it
def process_line(line):
    #read line to dictionary
    record = json.loads(line.strip())
    
    #remove fields that are not needed
    for field in unimportant_fields:
        record.pop(field)
    
    #convert text to lowercase
    record["text"] = record["text"].lower()
    
    #Remove special characters (Worse result when extra characters were removed)
    #record["text"] = re.sub(r'[^\w\s]', '', record["text"])
    
    return record


records = []
#read the datafile and process each line with process_line
with open(datafile, 'r', encoding='utf-8') as file:
    for line in file:
        records.append(process_line(line))
        
df = pd.DataFrame(records)

#split data into training and testing data
train, test = train_test_split(df, test_size=0.2, random_state=123)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

X_train_text = vectorizer.fit_transform(train['text'])
X_test_text = vectorizer.transform(test['text'])

y_train = train[['stars', 'useful', 'funny', 'cool']].values
y_test = test[['stars', 'useful', 'funny', 'cool']].values

print(f"Training data shape: {X_train_text.shape}")
print(f"Testing data shape: {X_test_text.shape}")

# Select 'stars' as the target
y_train_stars = y_train[:, 0]
y_test_stars = y_test[:, 0]

# Manual Naive Bayes Implementation
class NaiveBayes:
    def __init__(self):
        self.prior = {}
        self.likelihood = {}
        self.classes = []
        self.vocab_size = 0

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.vocab_size = X.shape[1]
        class_counts = {}
        word_counts = {}

        # Calculate prior probabilities and word counts
        for c in self.classes:
            indices = np.where(y == c)[0]
            class_counts[c] = len(indices)
            word_counts[c] = X[indices].sum(axis=0)

        total_samples = len(y)
        self.prior = {c: class_counts[c] / total_samples for c in self.classes}
        self.likelihood = {c: (word_counts[c] + 1) / (word_counts[c].sum() + self.vocab_size)
                           for c in self.classes}  # Laplace smoothing for zeros

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            log_probs = {}
            for c in self.classes:
                log_probs[c] = np.log(self.prior[c]) + (X[i].toarray() @ np.log(self.likelihood[c]).T).sum()
            predictions.append(max(log_probs, key=log_probs.get))
        return predictions

nb_model = NaiveBayes()
nb_model.fit(X_train_text, y_train_stars)

# Make predictions
y_pred = nb_model.predict(X_test_text)

report = classification_report(y_test_stars, y_pred, target_names=['1 star', '2 stars', '3 stars', '4 stars', '5 stars'])
print(report)