import argparse
import json
import os
from collections import Counter
import nltk
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight

nltk.download('punkt')
nltk.download('punkt_tab')

datafile = "reduced_data.json"
test_datafile = "test_data.jsonl"

# FOR CACHING TOKEN RESULTS
tokenized_cache_file = "processed_train_records_tokenized.json"
tokenized_test_cache_file = "processed_test_records_tokenized.json"

important_fields = ["stars", "useful", "funny", "cool", "text"]
unimportant_fields = ["review_id", "user_id", "business_id", "date"]

def process_line(line):
    record = json.loads(line.strip())
    
    for field in unimportant_fields:
        record.pop(field)
    
    record["text"] = record["text"].lower()
    
    #Remove special characters (Worse result when extra characters were removed)
    #record["text"] = re.sub(r'[^\w\s]', '', record["text"])
    
    return record

# Builds vocabulary of word tokens based on review text
def build_vocab(tokenized_texts, max_vocab_size=5000):
    word_counts = Counter()
    for tokens in tokenized_texts:
        word_counts.update(tokens)

    most_common_words = word_counts.most_common(max_vocab_size - 2)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common_words)}
    vocab['<PAD>'] = 0  # Padding token
    vocab['<UNK>'] = 1  # Unknown word token

    return vocab

# Function to convert text to indices
def tokenize_and_index(record, vocab):
    text = record['text']
    tokens = word_tokenize(text)
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    return indices


# Bins for each of the useful, funny, cool upvote categories.
def bin_upvote_counts(count):
    if count == 0:          # 0 Upvotes
        return 0
    elif 1 <= count <= 3:   # 1-3 Upvotes
        return 1
    elif 4 <= count <= 5:   # 4-5 Upvotes
        return 2
    else:
        return 3            # 6+ Upvotes 
    
def bin_stars(original_label):
    if original_label in [0, 1]:  # Combine 1-star and 2-star reviews
        return 0
    elif original_label == 2:     # 3-star reviews
        return 1
    elif original_label == 3:     # 4-star reviews
        return 2
    elif original_label == 4:     # 5-star reviews
        return 3

# Define the Dataset class
class YelpReviewDataset(Dataset):
    def __init__(self, records, vocab, max_seq_len=100):
        self.records = records
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]

        indices = tokenize_and_index(record, self.vocab)
        # Truncate or pad sequences
        if len(indices) > self.max_seq_len:
            indices = indices[:self.max_seq_len]
        else:
            indices += [self.vocab['<PAD>']] * (self.max_seq_len - len(indices))

        text_tensor = torch.tensor(indices, dtype=torch.long)

        stars_label = int(record['stars']) - 1
        stars_label = bin_stars(stars_label)

        useful_label = bin_upvote_counts(record['useful'])
        funny_label = bin_upvote_counts(record['funny'])
        cool_label = bin_upvote_counts(record['cool'])

        labels = torch.tensor([stars_label, useful_label, funny_label, cool_label], dtype=torch.long)
        return text_tensor, labels

# Create dataloader helpers
def create_dataloaders(train_records, val_records, vocab, batch_size=64):
    train_dataset = YelpReviewDataset(train_records, vocab)
    val_dataset = YelpReviewDataset(val_records, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


# Define the neural network model
class YelpSentimentNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes_stars, num_classes_useful, num_classes_funny, num_classes_cool):
        super(YelpSentimentNeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.fc_common = nn.Linear(embed_size, 64)
        self.dropout = nn.Dropout(0.5)
        # Output layers
        self.fc_stars = nn.Linear(64, num_classes_stars)
        self.fc_useful = nn.Linear(64, num_classes_useful)
        self.fc_funny = nn.Linear(64, num_classes_funny)
        self.fc_cool = nn.Linear(64, num_classes_cool)
    
    def forward(self, x):
        embeds = self.embedding(x)
        pooled = embeds.mean(dim=1)  
        out = F.relu(self.fc_common(pooled))
        out = self.dropout(out)

        # Outputs for each target
        stars_logits = self.fc_stars(out)
        useful_logits = self.fc_useful(out)
        funny_logits = self.fc_funny(out)
        cool_logits = self.fc_cool(out)
        return stars_logits, useful_logits, funny_logits, cool_logits

def get_upvote_class_weights(records,label_key):
    bins = [bin_upvote_counts(record[label_key]) for record in records]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2, 3]), y=bins)
    return torch.tensor(class_weights, dtype=torch.float)

# Training loop
def train_model(model, train_loader, val_loader, criterion_stars, criterion_useful, criterion_funny, criterion_cool, optimizer, num_epochs=5, device=torch.device('cpu')):
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        correct_stars = correct_useful = correct_funny = correct_cool = 0
        total = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            stars_logits, useful_logits, funny_logits, cool_logits = model(texts)
            
            loss_stars = criterion_stars(stars_logits, labels[:,0])
            loss_useful = criterion_useful(useful_logits, labels[:,1])
            loss_funny = criterion_funny(funny_logits, labels[:,2])
            loss_cool = criterion_cool(cool_logits, labels[:,3])
            
            loss = loss_stars + loss_useful + loss_funny + loss_cool
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            _, predicted_stars = torch.max(stars_logits.data, 1)
            _, predicted_useful = torch.max(useful_logits.data, 1)
            _, predicted_funny = torch.max(funny_logits.data, 1)
            _, predicted_cool = torch.max(cool_logits.data, 1)
            total += labels.size(0)
            correct_stars += (predicted_stars == labels[:,0]).sum().item()
            correct_useful += (predicted_useful == labels[:,1]).sum().item()
            correct_funny += (predicted_funny == labels[:,2]).sum().item()
            correct_cool += (predicted_cool == labels[:,3]).sum().item()
        
        avg_train_loss = sum(train_losses)/len(train_losses)
        train_acc_stars = correct_stars / total
        train_acc_useful = correct_useful / total
        train_acc_funny = correct_funny / total
        train_acc_cool = correct_cool / total

        # Validation
        model.eval()
        val_losses = []
        val_correct_stars = val_correct_useful = val_correct_funny = val_correct_cool = 0
        val_total = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                stars_logits, useful_logits, funny_logits, cool_logits = model(texts)
                
                loss_stars = criterion_stars(stars_logits, labels[:,0])
                loss_useful = criterion_useful(useful_logits, labels[:,1])
                loss_funny = criterion_funny(funny_logits, labels[:,2])
                loss_cool = criterion_cool(cool_logits, labels[:,3])

                loss = loss_stars + loss_useful + loss_funny + loss_cool
                val_losses.append(loss.item())

                _, predicted_stars = torch.max(stars_logits.data, 1)
                _, predicted_useful = torch.max(useful_logits.data, 1)
                _, predicted_funny = torch.max(funny_logits.data, 1)
                _, predicted_cool = torch.max(cool_logits.data, 1)
                val_total += labels.size(0)
                val_correct_stars += (predicted_stars == labels[:,0]).sum().item()
                val_correct_useful += (predicted_useful == labels[:,1]).sum().item()
                val_correct_funny += (predicted_funny == labels[:,2]).sum().item()
                val_correct_cool += (predicted_cool == labels[:,3]).sum().item()

        avg_val_loss = sum(val_losses)/len(val_losses)
        val_acc_stars = val_correct_stars / val_total
        val_acc_useful = val_correct_useful / val_total
        val_acc_funny = val_correct_funny / val_total
        val_acc_cool = val_correct_cool / val_total

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')
        print(f'Train Acc - Stars: {train_acc_stars:.4f}, Useful: {train_acc_useful:.4f}, Funny: {train_acc_funny:.4f}, Cool: {train_acc_cool:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Acc - Stars: {val_acc_stars:.4f}, Useful: {val_acc_useful:.4f}, Funny: {val_acc_funny:.4f}, Cool: {val_acc_cool:.4f}')
        print('-'*80)



# Evaluate the model
def evaluate_model_with_classification_summary(model, data_loader, device):
    model.eval()
    stars_predictions, stars_actuals = [], []
    useful_predictions, useful_actuals = [], []
    funny_predictions, funny_actuals = [], []
    cool_predictions, cool_actuals = [], []

    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            stars_logits, useful_logits, funny_logits, cool_logits = model(texts)
            _, predicted_stars = torch.max(stars_logits.data, 1)
            _, predicted_useful = torch.max(useful_logits.data, 1)
            _, predicted_funny = torch.max(funny_logits.data, 1)
            _, predicted_cool = torch.max(cool_logits.data, 1)
            
            stars_predictions.extend(predicted_stars.cpu().numpy())
            stars_actuals.extend(labels[:, 0].cpu().numpy())
            useful_predictions.extend(predicted_useful.cpu().numpy())
            useful_actuals.extend(labels[:, 1].cpu().numpy())
            funny_predictions.extend(predicted_funny.cpu().numpy())
            funny_actuals.extend(labels[:, 2].cpu().numpy())
            cool_predictions.extend(predicted_cool.cpu().numpy())
            cool_actuals.extend(labels[:, 3].cpu().numpy())

    star_targets = ['1-2','3','4','5']
    upvote_bin_targets = ['0', '1-3', '4-5', '6+']

    print("\nStars Classification Report ---")
    print(classification_report(stars_actuals, stars_predictions, target_names=star_targets, zero_division=0))
    print("\nUseful Classification Report ---")
    print(classification_report(useful_actuals, useful_predictions, target_names=upvote_bin_targets, zero_division=0))
    print("\nFunny Classification Report ---")
    print(classification_report(funny_actuals, funny_predictions, target_names=upvote_bin_targets, zero_division=0))
    print("\nCool Classification Report ---")
    print(classification_report(cool_actuals, cool_predictions, target_names=upvote_bin_targets, zero_division=0))

    # Calculate and print MSE for each target
    stars_mse = mean_squared_error(stars_actuals, stars_predictions)
    useful_mse = mean_squared_error(useful_actuals, useful_predictions)
    funny_mse = mean_squared_error(funny_actuals, funny_predictions)
    cool_mse = mean_squared_error(cool_actuals, cool_predictions)

    print("\nMean Squared Errors: ---")
    print(f"Stars MSE: {stars_mse:.4f}")
    print(f"Useful MSE: {useful_mse:.4f}")
    print(f"Funny MSE: {funny_mse:.4f}")
    print(f"Cool MSE: {cool_mse:.4f}")



parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action='store_true', help='Use GPU acceleration if available')
parser.add_argument('--cache', action='store_true', help='Use cached tokenized data if available')
args = parser.parse_args()

if args.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Using device:", device)

start_time = time.time()

# Training
if args.cache and os.path.exists(tokenized_cache_file):
    print("Loading TRAINING Cache JSON")
    with open(tokenized_cache_file, 'r', encoding='utf-8') as f:
        processed_train_records = json.load(f)
else:
    processed_train_records = []
    with open(datafile, 'r', encoding='utf-8') as f:
        for line in f:
            record = process_line(line)
            if all(k in record for k in ['stars', 'useful', 'funny', 'cool']):
                record["tokens"] = word_tokenize(record["text"])
                processed_train_records.append(record)
    if args.cache:
        print("Saving training data to JSON Cache")
        with open(tokenized_cache_file, 'w', encoding='utf-8') as out_f:
            json.dump(processed_train_records, out_f)


# Test
if args.cache and os.path.exists(tokenized_test_cache_file):
    print("Loading TEST Cache JSON")
    with open(tokenized_test_cache_file, 'r', encoding='utf-8') as f:
        processed_test_records = json.load(f)
else:
    processed_test_records = []
    with open(test_datafile, 'r', encoding='utf-8') as f:
        for line in f:
            record = process_line(line)
            if all(k in record for k in ['stars', 'useful', 'funny', 'cool']):
                record["tokens"] = word_tokenize(record["text"])
                processed_test_records.append(record)
    # Optionally save to cache for next time
    if args.cache:
        print("Saving test data to cached JSON")
        with open(tokenized_test_cache_file, 'w', encoding='utf-8') as out_f:
            json.dump(processed_test_records, out_f)

tokenized_texts = [word_tokenize(record['text']) for record in processed_train_records]
vocab = build_vocab(tokenized_texts, max_vocab_size=5000)

vocab_size = len(vocab)
embed_size = 128  
num_classes_stars = 4   
num_classes_useful = 4  
num_classes_funny = 4
num_classes_cool = 4

neuralNetworkModel = YelpSentimentNeuralNetwork(vocab_size, embed_size, num_classes_stars, num_classes_useful, num_classes_funny, num_classes_cool)
neuralNetworkModel.to(device)

useful_weights = get_upvote_class_weights(processed_train_records, "useful").to(device)
funny_weights = get_upvote_class_weights(processed_train_records, "funny").to(device)
cool_weights = get_upvote_class_weights(processed_train_records, "cool").to(device)

criterion_stars = nn.CrossEntropyLoss()
criterion_useful = nn.CrossEntropyLoss(weight=useful_weights)
criterion_funny = nn.CrossEntropyLoss(weight=funny_weights)
criterion_cool = nn.CrossEntropyLoss(weight=cool_weights)

optimizer = torch.optim.Adam(neuralNetworkModel.parameters(), lr=0.001)

train_records, val_records = train_test_split(processed_train_records, test_size=0.2, random_state=42)
train_loader, val_loader = create_dataloaders(train_records, val_records, vocab, batch_size=64)

test_dataset = YelpReviewDataset(processed_test_records, vocab)
test_loader = DataLoader(test_dataset, batch_size=64)

train_model(neuralNetworkModel, train_loader, val_loader, criterion_stars, criterion_useful, criterion_funny, criterion_cool, optimizer, num_epochs=5, device=device)
evaluate_model_with_classification_summary(neuralNetworkModel, test_loader, device)

end_time = time.time()
elapsed_time = end_time - start_time

print (f"\n Start Time: {start_time:.2f}")
print (f"\n End Time: {end_time:.2f}")
print (f"Total runtime: {elapsed_time:.2f} seconds")