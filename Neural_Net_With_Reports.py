import json
from collections import Counter
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt')
nltk.download('punkt_tab')

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

processed_records = []
with open('yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
    for line in f:
        record = process_line(line)

        if all(k in record for k in ['stars', 'useful', 'funny', 'cool']):
            processed_records.append(record)

        if len(processed_records) >= 10000:
            break

# Build the vocabulary
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
def bin_count(count):
    if count == 0:          # 0 Upvotes
        return 0
    elif 1 <= count <= 3:   # 1-3 Upvotes
        return 1
    elif 4 <= count <= 5:   # 4-5 Upvotes
        return 2
    else:
        return 3            # 6+ Upvotes 

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
        # Prepare labels
        # For 'stars', adjust to zero-based indexing for classification (labels from 0 to 4)
        stars_label = int(record['stars']) - 1  # Adjust stars to 0-4
        # Classification labels for 'useful', 'funny', 'cool'
        useful_label = bin_count(record['useful'])
        funny_label = bin_count(record['funny'])
        cool_label = bin_count(record['cool'])
        # Combine labels into a tensor
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
        # x shape: (batch_size, max_seq_len)
        embeds = self.embedding(x)  # (batch_size, max_seq_len, embed_size)
        # Average pooling over sequence length
        pooled = embeds.mean(dim=1)  # (batch_size, embed_size)
        out = F.relu(self.fc_common(pooled))
        out = self.dropout(out)
        # Outputs for each target
        stars_logits = self.fc_stars(out)
        useful_logits = self.fc_useful(out)
        funny_logits = self.fc_funny(out)
        cool_logits = self.fc_cool(out)
        return stars_logits, useful_logits, funny_logits, cool_logits

# Instantiate the model


# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        correct_stars = 0
        correct_useful = 0
        correct_funny = 0
        correct_cool = 0
        total = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            stars_logits, useful_logits, funny_logits, cool_logits = model(texts)
            # Compute losses
            loss_stars = criterion(stars_logits, labels[:, 0])
            loss_useful = criterion(useful_logits, labels[:, 1])
            loss_funny = criterion(funny_logits, labels[:, 2])
            loss_cool = criterion(cool_logits, labels[:, 3])
            loss = loss_stars + loss_useful + loss_funny + loss_cool
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            # Compute training accuracy
            _, predicted_stars = torch.max(stars_logits.data, 1)
            _, predicted_useful = torch.max(useful_logits.data, 1)
            _, predicted_funny = torch.max(funny_logits.data, 1)
            _, predicted_cool = torch.max(cool_logits.data, 1)
            total += labels.size(0)
            correct_stars += (predicted_stars == labels[:, 0]).sum().item()
            correct_useful += (predicted_useful == labels[:, 1]).sum().item()
            correct_funny += (predicted_funny == labels[:, 2]).sum().item()
            correct_cool += (predicted_cool == labels[:, 3]).sum().item()
        avg_train_loss = sum(train_losses) / len(train_losses)
        train_acc_stars = correct_stars / total
        train_acc_useful = correct_useful / total
        train_acc_funny = correct_funny / total
        train_acc_cool = correct_cool / total
        
        # Validation
        model.eval()
        val_losses = []
        val_correct_stars = 0
        val_correct_useful = 0
        val_correct_funny = 0
        val_correct_cool = 0
        val_total = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                stars_logits, useful_logits, funny_logits, cool_logits = model(texts)
                # Compute losses
                loss_stars = criterion(stars_logits, labels[:, 0])
                loss_useful = criterion(useful_logits, labels[:, 1])
                loss_funny = criterion(funny_logits, labels[:, 2])
                loss_cool = criterion(cool_logits, labels[:, 3])
                loss = loss_stars + loss_useful + loss_funny + loss_cool
                val_losses.append(loss.item())
                # Compute validation accuracy
                _, predicted_stars = torch.max(stars_logits.data, 1)
                _, predicted_useful = torch.max(useful_logits.data, 1)
                _, predicted_funny = torch.max(funny_logits.data, 1)
                _, predicted_cool = torch.max(cool_logits.data, 1)
                val_total += labels.size(0)
                val_correct_stars += (predicted_stars == labels[:, 0]).sum().item()
                val_correct_useful += (predicted_useful == labels[:, 1]).sum().item()
                val_correct_funny += (predicted_funny == labels[:, 2]).sum().item()
                val_correct_cool += (predicted_cool == labels[:, 3]).sum().item()
        avg_val_loss = sum(val_losses) / len(val_losses)
        val_acc_stars = val_correct_stars / val_total
        val_acc_useful = val_correct_useful / val_total
        val_acc_funny = val_correct_funny / val_total
        val_acc_cool = val_correct_cool / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')
        print(f'Train Acc - Stars: {train_acc_stars:.4f}, Useful: {train_acc_useful:.4f}, Funny: {train_acc_funny:.4f}, Cool: {train_acc_cool:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Acc - Stars: {val_acc_stars:.4f}, Useful: {val_acc_useful:.4f}, Funny: {val_acc_funny:.4f}, Cool: {val_acc_cool:.4f}')
        print('-' * 80)


# Evaluate the model
def evaluate_model_with_classification_summary(model, data_loader):
    model.eval()
    stars_predictions, stars_actuals = [], []
    useful_predictions, useful_actuals = [], []
    funny_predictions, funny_actuals = [], []
    cool_predictions, cool_actuals = [], []

    with torch.no_grad():
        for texts, labels in data_loader:
            stars_logits, useful_logits, funny_logits, cool_logits = model(texts)
            # Predictions for each output
            _, predicted_stars = torch.max(stars_logits.data, 1)
            _, predicted_useful = torch.max(useful_logits.data, 1)
            _, predicted_funny = torch.max(funny_logits.data, 1)
            _, predicted_cool = torch.max(cool_logits.data, 1)
            
            # Collect predictions and actuals
            stars_predictions.extend(predicted_stars.numpy())
            stars_actuals.extend(labels[:, 0].numpy())
            useful_predictions.extend(predicted_useful.numpy())
            useful_actuals.extend(labels[:, 1].numpy())
            funny_predictions.extend(predicted_funny.numpy())
            funny_actuals.extend(labels[:, 2].numpy())
            cool_predictions.extend(predicted_cool.numpy())
            cool_actuals.extend(labels[:, 3].numpy())

    star_targets = ['1','2','3','4','5']
    upvote_bin_targets = ['0', '1-3', '4-5', '6+']

    # Print detailed classification reports
    print("\nStars Classification Report ---")
    print(classification_report(stars_actuals, stars_predictions, target_names=star_targets, zero_division=0))
    print("\nUseful Classification Report ---")
    print(classification_report(useful_actuals, useful_predictions, target_names=upvote_bin_targets, zero_division=0))
    print("\nFunny Classification Report ---")
    print(classification_report(funny_actuals, funny_predictions, target_names=upvote_bin_targets, zero_division=0))
    print("\nCool Classification Report ---")
    print(classification_report(cool_actuals, cool_predictions, target_names=upvote_bin_targets, zero_division=0))


tokenized_texts = [word_tokenize(record['text']) for record in processed_records]
vocab = build_vocab(tokenized_texts, max_vocab_size=5000)

vocab_size = len(vocab)
embed_size = 128  
num_classes_stars = 5   
num_classes_useful = 4  
num_classes_funny = 4
num_classes_cool = 4

neuralNetworkModel = YelpSentimentNeuralNetwork(vocab_size, embed_size, num_classes_stars, num_classes_useful, num_classes_funny, num_classes_cool)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neuralNetworkModel.parameters(), lr=0.001)


train_records, val_records = train_test_split(processed_records, test_size=0.2, random_state=42)
train_loader, val_loader = create_dataloaders(train_records, val_records, vocab, batch_size=64)
train_model(neuralNetworkModel, train_loader, val_loader, criterion, optimizer, num_epochs=5)
evaluate_model_with_classification_summary(neuralNetworkModel, val_loader)
