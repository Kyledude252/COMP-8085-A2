import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import re

datafile = "reduced_data.json"
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
y_train_stars = y_train[:, 0]  # First column corresponds to 'stars'
y_test_stars = y_test[:, 0]

# Train a Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_text, y_train_stars)

# Make predictions
y_pred = nb_model.predict(X_test_text)
y_true = y_test[:, 0]  # Actual 'stars' values

report = classification_report(y_true, y_pred, target_names=['1 star', '2 stars', '3 stars', '4 stars', '5 stars'])
print(report)

# Calculate exact-match accuracy
#exact_accuracy = (y_pred == y_true).mean() * 100
#print(f"Exact Match Accuracy for 'stars': {exact_accuracy:.2f}%")

# Evaluate the model
#mse = mean_squared_error(y_test_stars, y_pred)
#print(f"Mean Squared Error for 'stars': {mse}")

#print(train.head())
#print(test.head())