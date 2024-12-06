import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import re
import numpy as np
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

datafile = "reduced_data.json"
#datafile = "yelp_academic_dataset_review.json"
important_fields = ["stars", "useful", "funny", "cool", "text"]
unimportant_fields = ["review_id", "user_id", "business_id", "date"]

# **** DATA PROCESSING ****
#process a single line of the json data and return it
def process_line(line):
    #read line to dictionary
    record = json.loads(line.strip())
    
    #remove fields that are not needed
    for field in unimportant_fields:
        record.pop(field)
    
    #convert text to lowercase
    record["text"] = record["text"].lower()
    
    #Remove special characters
    record["text"] = re.sub(r'[^\w\s]', '', record["text"])
    
    #Remove stopwords
    stop_words = set(stopwords.words('english'))
    record["text"] = ' '.join([word for word in record["text"].split() if word not in stop_words])
    
    return record


records = []
#read the datafile and process each line with process_line
with open(datafile, 'r', encoding='utf-8') as file:
    for line in file:
        records.append(process_line(line))
        
df = pd.DataFrame(records)

print("Dataset Saved")

#split data into training and testing data
train, test = train_test_split(df, test_size=0.2, random_state=123)

# **** VECTORIZER ****
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

X_train_text = vectorizer.fit_transform(train['text'])
X_test_text = vectorizer.transform(test['text'])

y_train = train[['stars', 'useful', 'funny', 'cool']].values
y_test = test[['stars', 'useful', 'funny', 'cool']].values

print(f"Training data shape: {X_train_text.shape}")
print(f"Testing data shape: {X_test_text.shape}")

# Select targets
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

# list of target fields for text to classify
target_fields = ["stars", "useful", "funny", "cool"]

def group_classes(y):
    """
    Maps all classes greater than 5 to 6.
    """
    y = np.array(y)
    y = np.where(y > 5, 6, y)
    return y

# process reports
for i, field in enumerate(target_fields):
    print(f"\n--- Classification Report for '{field}' ---")
    
    # Select the target column and group classes
    y_train_field = group_classes(y_train[:, i])
    y_test_field = group_classes(y_test[:, i])
    
    # Train a new Naive Bayes model
    nb_model = NaiveBayes()
    nb_model.fit(X_train_text, y_train_field)
    
    # Make predictions
    y_pred_field = nb_model.predict(X_test_text)
    
    # Generate and print the classification report
    unique_labels = np.unique(y_test_field)
    target_names = [f"{int(label)}" for label in unique_labels]
    
    # Rename the last label to "6+" if it exists
    if "6" in target_names:
        target_names[-1] = "6+"
    
    report = classification_report(
        y_test_field, y_pred_field, target_names=target_names, zero_division=0
    )
    print(report)