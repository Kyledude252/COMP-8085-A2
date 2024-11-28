import pandas as pd
import json
from sklearn.model_selection import train_test_split

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
    
    return record


records = []
#read the datafile and process each line with process_line
with open(datafile, 'r', encoding='utf-8') as file:
    for line in file:
        records.append(process_line(line))
        
df = pd.DataFrame(records)

#split data into training and testing data
train, test = train_test_split(df, test_size=0.2, random_state=123)

print(train.head())
print(test.head())