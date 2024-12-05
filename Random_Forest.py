import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import mean_squared_error, classification_report
import re
from joblib import dump, load

datafile = "reduced_data.json"
model_path = "random_forest_model.joblib"
vectorizer_path = "random_forest_vectorizer.joblib"
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

def train():

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

    # Save the trained model and vectorizer
    dump(multi_rf, model_path)
    dump(vectorizer, vectorizer_path)
    print("Model and vectorizer saved successfully.")

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
    

# Inference mode
def inference():
    from sklearn.metrics import classification_report

    # Load model and vectorizer
    print("Loading model and vectorizer...")
    multi_rf = load(model_path)
    vectorizer = load(vectorizer_path)

    # Read and process inference data
    records = []
    with open("test_data.json", 'r', encoding='utf-8') as file:
        for line in file:
            records.append(process_line(line))
    df = pd.DataFrame(records)

    # Ensure the required fields exist for classification report
    if not {"stars", "useful", "funny", "cool"}.issubset(df.columns):
        raise ValueError("Inference data must include 'stars', 'useful', 'funny', and 'cool' fields for a classification report.")

    # Vectorize text data
    X_inference = vectorizer.transform(df['text'])

    # Make predictions
    print("Making predictions...")
    y_pred = multi_rf.predict(X_inference)

    # Combine predictions with input data
    output_df = pd.concat([df[['text', 'stars', 'useful', 'funny', 'cool']], 
                           pd.DataFrame(y_pred, columns=["predicted_stars", "predicted_useful", "predicted_funny", "predicted_cool"])], axis=1)

    # Generate classification report for each target
    print("\nClassification Reports:")
    for target in ["stars", "useful", "funny", "cool"]:
        true_labels = df[target]
        predicted_labels = output_df[f"predicted_{target}"]
        print(f"\n{target.capitalize()} Classification Report:")
        print(classification_report(true_labels, predicted_labels, zero_division=0))

    
    
if __name__ == "__main__":
    mode = input("Type 'T' for Training mode or 'I' for inference mode\n").strip().lower()
    if mode == "T" or mode == "t":
        train()
    elif mode == "I" or mode == "i":
        inference()
    else:
        print("Invalid mode. Please enter 'T' or 'I'.")