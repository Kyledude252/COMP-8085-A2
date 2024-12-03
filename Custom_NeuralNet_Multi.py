import json
import re
import string
from collections import Counter
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.metrics import mean_squared_error

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Define unimportant fields to remove
unimportant_fields = ['review_id', 'user_id', 'business_id', 'date']

# Process a single line of the JSON data and return it
def process_line(line):
    # Read line to dictionary
    record = json.loads(line.strip())
    
    # Remove fields that are not needed
    for field in unimportant_fields:
        record.pop(field, None)  # Use pop with default to avoid KeyError if field is missing
    
    # Convert text to lowercase
    record["text"] = record["text"].lower()
    
    # Remove special characters (optional)
    # record["text"] = re.sub(r'[^\w\s]', '', record["text"])
    
    return record

# Read and preprocess the data
processed_records = []
with open('yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
    for line in f:
        record = process_line(line)
        # Check if all required labels are present
        if all(k in record for k in ['stars', 'useful', 'funny', 'cool']):
            processed_records.append(record)
        # Limit the number of records for faster training (e.g., first 10,000 reviews)
        if len(processed_records) >= 50000:
            break

# Tokenize texts
tokenized_texts = [word_tokenize(record['text']) for record in processed_records]

# Build the vocabulary
def build_vocab(tokenized_texts, max_vocab_size=5000):
    word_counts = Counter()
    for tokens in tokenized_texts:
        word_counts.update(tokens)
    # Keep the most common words
    most_common_words = word_counts.most_common(max_vocab_size - 2)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common_words)}
    vocab['<PAD>'] = 0  # Padding token
    vocab['<UNK>'] = 1  # Unknown word token
    return vocab

vocab = build_vocab(tokenized_texts, max_vocab_size=5000)

# Function to convert text to indices
def tokenize_and_index(record, vocab):
    text = record['text']
    tokens = word_tokenize(text)
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    return indices

# Split into training and validation sets
train_records, val_records = train_test_split(processed_records, test_size=0.2, random_state=42)

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
        # Convert text to indices
        indices = tokenize_and_index(record, self.vocab)
        # Truncate or pad sequences
        if len(indices) > self.max_seq_len:
            indices = indices[:self.max_seq_len]
        else:
            indices += [self.vocab['<PAD>']] * (self.max_seq_len - len(indices))
        # Convert to tensor
        text_tensor = torch.tensor(indices, dtype=torch.long)
        # Get labels (assuming all labels are present)
        labels = torch.tensor([
            record['stars'],
            record['useful'],
            record['funny'],
            record['cool']
        ], dtype=torch.float)
        return text_tensor, labels

# Create DataLoaders
def create_dataloaders(train_records, val_records, vocab, batch_size=64):
    train_dataset = YelpReviewDataset(train_records, vocab)
    val_dataset = YelpReviewDataset(val_records, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

train_loader, val_loader = create_dataloaders(train_records, val_records, vocab, batch_size=64)

# Define the neural network model
class MultiOutputNN(nn.Module):
    def __init__(self, vocab_size, embed_size, output_size):
        super(MultiOutputNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.fc1 = nn.Linear(embed_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x shape: (batch_size, max_seq_len)
        embeds = self.embedding(x)  # (batch_size, max_seq_len, embed_size)
        # Average pooling over sequence length
        pooled = embeds.mean(dim=1)  # (batch_size, embed_size)
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        out = self.fc2(out)  # (batch_size, output_size)
        return out  # No need to squeeze since output is (batch_size, output_size)

# Instantiate the model
vocab_size = len(vocab)
embed_size = 128  # You can adjust this value
output_size = 4   # Predicting four values: stars, useful, funny, cool

model = MultiOutputNN(vocab_size, embed_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for texts, labels in val_loader:
                outputs = model(texts)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

# Evaluate the model
def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for texts, labels in data_loader:
            outputs = model(texts)
            predictions.extend(outputs.numpy())
            actuals.extend(labels.numpy())
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    # Calculate MSE for each output
    mse_values = ((predictions - actuals) ** 2).mean(axis=0)
    print(f'MSE on validation set: Stars: {mse_values[0]:.4f}, Useful: {mse_values[1]:.4f}, Funny: {mse_values[2]:.4f}, Cool: {mse_values[3]:.4f}')

# Import numpy for evaluation
import numpy as np

# Run evaluation
evaluate_model(model, val_loader)
