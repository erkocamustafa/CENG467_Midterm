"""
Q1: Text Classification & Representation Learning
Dataset: IMDb (Local CSVs)
Models: TF-IDF + SVM, PyTorch BiLSTM, DistilBERT
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Q1] Running on: {device}")

# 1. DATA LOADING & PREPROCESSING
print("Loading IMDb dataset...")
train_full = pd.read_csv("datasets/imdb/train.csv")
test_full = pd.read_csv("datasets/imdb/test.csv")

# Sample
train_df = train_full.sample(n=5000, random_state=42).reset_index(drop=True)
test_df = test_full.sample(n=2000, random_state=42).reset_index(drop=True)

# Split Validation
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Load Spacy for linguistic cleaning
try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
except OSError:
    print("Downloading spacy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def clean_text(text, remove_stops=True):
    """Regex cleaning and Spacy lemmatization"""
    text = re.sub(r'<[^>]+>', ' ', text) # Remove HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower() # Keep only letters
    
    doc = nlp(text)
    if remove_stops:
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
    else:
        tokens = [token.lemma_ for token in doc if not token.is_space]
    return " ".join(tokens)

print("Preprocessing text (this takes a moment)...")
train_df['clean_text'] = train_df['text'].apply(lambda x: clean_text(x, True))
val_df['clean_text'] = val_df['text'].apply(lambda x: clean_text(x, True))
test_df['clean_text'] = test_df['text'].apply(lambda x: clean_text(x, True))


# MODEL 1: TF-IDF + SVM (Sparse Representation)
print("\n--- Training Model 1: TF-IDF + LinearSVC ---")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

X_train_tfidf = vectorizer.fit_transform(train_df['clean_text'])
X_test_tfidf = vectorizer.transform(test_df['clean_text'])

svm_model = LinearSVC(random_state=42, dual="auto")
svm_model.fit(X_train_tfidf, train_df['label'])

svm_preds = svm_model.predict(X_test_tfidf)
svm_acc = accuracy_score(test_df['label'], svm_preds)
svm_f1 = f1_score(test_df['label'], svm_preds, average='macro')
print(f"SVM Accuracy: {svm_acc:.4f} | Macro-F1: {svm_f1:.4f}")


# MODEL 2: PyTorch BiLSTM (Dense Representation)
print("\n--- Training Model 2: PyTorch BiLSTM ---")

# Build Vocab
all_words = " ".join(train_df['clean_text']).split()
vocab = {word: i+2 for i, word in enumerate(set(all_words))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

def encode_sentence(text, max_len=200):
    tokens = text.split()
    encoded = [vocab.get(w, 1) for w in tokens][:max_len]
    padding = [0] * (max_len - len(encoded))
    return encoded + padding

# Prepare Tensors
X_train_seq = torch.tensor([encode_sentence(t) for t in train_df['clean_text']])
y_train_seq = torch.tensor(train_df['label'].values, dtype=torch.float32)
X_test_seq = torch.tensor([encode_sentence(t) for t in test_df['clean_text']])

train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=32, shuffle=True)

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # Concat final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.sigmoid(self.fc(hidden_cat)).squeeze()

lstm_model = BiLSTM(len(vocab), embed_dim=128, hidden_dim=64).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

lstm_model.train()
for epoch in range(3):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        predictions = lstm_model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

lstm_model.eval()
with torch.no_grad():
    X_test_seq = X_test_seq.to(device)
    lstm_probs = lstm_model(X_test_seq).cpu().numpy()
    lstm_preds = (lstm_probs > 0.5).astype(int)

lstm_acc = accuracy_score(test_df['label'], lstm_preds)
lstm_f1 = f1_score(test_df['label'], lstm_preds, average='macro')
print(f"BiLSTM Accuracy: {lstm_acc:.4f} | Macro-F1: {lstm_f1:.4f}")

# Free GPU memory before loading Transformer
del lstm_model
torch.cuda.empty_cache() 


# MODEL 3: DistilBERT (Contextual Representation)
print("\n--- Training Model 3: DistilBERT ---")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

# Tokenize Data
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')

bert_train_loader = DataLoader(TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_df['label'].values)), batch_size=16, shuffle=True)
bert_test_loader = DataLoader(TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_df['label'].values)), batch_size=32)

bert_optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5)

bert_model.train()
for epoch in range(2): # for BERT fine-tuning
    for ids, mask, labels in bert_train_loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        bert_optimizer.zero_grad()
        outputs = bert_model(ids, attention_mask=mask, labels=labels)
        outputs.loss.backward()
        bert_optimizer.step()

bert_model.eval()
bert_preds = []
with torch.no_grad():
    for ids, mask, _ in bert_test_loader:
        outputs = bert_model(ids.to(device), attention_mask=mask.to(device))
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        bert_preds.extend(preds)

bert_acc = accuracy_score(test_df['label'], bert_preds)
bert_f1 = f1_score(test_df['label'], bert_preds, average='macro')
print(f"DistilBERT Accuracy: {bert_acc:.4f} | Macro-F1: {bert_f1:.4f}")

# ERROR ANALYSIS
print("\n--- Error Analysis (BERT Misclassifications) ---")
errors = np.where(bert_preds != test_df['label'].values)[0]
for i in range(min(5, len(errors))):
    idx = errors[i]
    print(f"\n[True: {test_df['label'].iloc[idx]} | Pred: {bert_preds[idx]}]")
    print(test_df['text'].iloc[idx][:250] + "...")