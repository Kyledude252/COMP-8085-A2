import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import mean_squared_error, classification_report
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

# Split data into training and testing data
train, test = train_test_split(df, test_size=0.2, random_state=123)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_text = vectorizer.fit_transform(train['text'])
X_test_text = vectorizer.transform(test['text'])

# Ensure y_train and y_test remain as DataFrames
y_train = train[['stars', 'useful', 'funny', 'cool']]
y_test = test[['stars', 'useful', 'funny', 'cool']]

print(f"Training data shape: {X_train_text.shape}")
print(f"Testing data shape: {X_test_text.shape}")

# Train a single model using MultiOutputClassifier
print("Training MultiOutput RandomForestClassifier...")
multi_rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=123, n_jobs=-1))
multi_rf.fit(X_train_text, y_train)

# Make predictions
y_pred = multi_rf.predict(X_test_text)

# Convert predictions to a DataFrame for easier handling
y_pred_df = pd.DataFrame(y_pred, columns=['stars', 'useful', 'funny', 'cool'], index=test.index)

# Calculate Mean Squared Errors
mse_scores = {}
for column in y_train.columns:  # Now y_train.columns will work
    mse_scores[column] = mean_squared_error(y_test[column], y_pred_df[column])

# Print Mean Squared Errors
print("\nMean Squared Errors:")
for target, mse in mse_scores.items():
    print(f"{target}: {mse}")

# Print Classification Reports
print("\nClassification Reports:")
for column in y_train.columns:
    print(f"\n{column.capitalize()} Classification Report:")
    print(classification_report(y_test[column], y_pred_df[column], zero_division=0))