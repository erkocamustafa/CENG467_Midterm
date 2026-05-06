"""
CENG 467 - Q2: Named Entity Recognition
Dataset: CoNLL-2003
Models: DistilBERT and BiLSTM-CRF
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.optim import AdamW
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
from pathlib import Path
from torchcrf import CRF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Q2] Running on: {device}")

# 1. PARSE LOCAL CONLL DATA
def parse_conll(filepath):
    """Extracts sentences and their BIO tags"""
    sentences, tags = [], []
    current_sent, current_tags = [], []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-") or not line:
                if current_sent:
                    sentences.append(current_sent)
                    tags.append(current_tags)
                    current_sent, current_tags = [], []
            else:
                splits = line.split()
                current_sent.append(splits[0])
                current_tags.append(splits[-1])
    return sentences, tags

data_dir = Path("datasets/conll2003")
train_sents, train_tags = parse_conll(data_dir / "train.txt")
test_sents, test_tags = parse_conll(data_dir / "test.txt")

unique_tags = set(tag for doc in train_tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

# 2. TOKEN ALIGNMENT & DATASET (DistilBERT)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

def align_labels(texts, labels):
    tokenized_inputs = tokenizer(texts, truncation=True, is_split_into_words=True, padding=True, return_tensors="pt")
    
    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) 
            elif word_idx != previous_word_idx:
                label_ids.append(tag2id[label[word_idx]])
            else:
                label_ids.append(-100) 
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
        
    tokenized_inputs["labels"] = torch.tensor(aligned_labels)
    return tokenized_inputs

print("Aligning Subwords... (Takes a minute)")
# Subset to save time as requested
train_encodings = align_labels(train_sents[:3000], train_tags[:3000]) 
test_encodings = align_labels(test_sents[:1000], test_tags[:1000])

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

train_loader = DataLoader(NERDataset(train_encodings), batch_size=16, shuffle=True)
test_loader = DataLoader(NERDataset(test_encodings), batch_size=16)

# 3. TRAIN & EVALUATE TRANSFORMER (DistilBERT)
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-cased", num_labels=len(tag2id)).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

print("\n--- Training DistilBERT for NER ---")
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.3f}")

print("\n--- Evaluating DistilBERT on Test Set ---")
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()
        pred_tags = np.argmax(logits, axis=2)
        
        for i in range(len(labels)):
            pred_seq = [id2tag[p] for (p, l) in zip(pred_tags[i], labels[i]) if l != -100]
            true_seq = [id2tag[l] for (p, l) in zip(pred_tags[i], labels[i]) if l != -100]
            predictions.append(pred_seq)
            true_labels.append(true_seq)

print(classification_report(true_labels, predictions))

# 4. DATA PREP FOR BiLSTM-CRF
print("\n--- Preparing Data for BiLSTM-CRF ---")
all_words = set(word for sent in train_sents[:3000] for word in sent)
word2id = {word: i for i, word in enumerate(all_words)}
word2id['<PAD>'] = len(word2id)
word2id['<UNK>'] = len(word2id)
vocab_size = len(word2id)

def encode_sentences(sentences, word2id):
    return [[word2id.get(word, word2id['<UNK>']) for word in sent] for sent in sentences]

train_word_ids = encode_sentences(train_sents[:3000], word2id)
test_word_ids = encode_sentences(test_sents[:1000], word2id)

def filter_empty_sequences(word_ids, labels):
    filtered_ids, filtered_labels = [], []
    for sent_ids, tag_seq in zip(word_ids, labels):
        if len(sent_ids) > 0:
            filtered_ids.append(sent_ids)
            filtered_labels.append(tag_seq)
    return filtered_ids, filtered_labels

train_word_ids, train_tags_filtered = filter_empty_sequences(train_word_ids, train_tags[:3000])
test_word_ids, test_tags_filtered = filter_empty_sequences(test_word_ids, test_tags[:1000])

class BiLSTMDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = [torch.tensor([tag2id[tag] for tag in lab]) for lab in labels]
    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]
    def __len__(self):
        return len(self.sentences)

def collate_fn(batch):
    sentences, labels = zip(*batch)
    sentences_padded = nn.utils.rnn.pad_sequence([torch.tensor(s) for s in sentences], batch_first=True, padding_value=word2id['<PAD>'])
    pad_label = tag2id.get('O', 0)
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_label)
    return sentences_padded, labels_padded

train_bilstm_loader = DataLoader(BiLSTMDataset(train_word_ids, train_tags_filtered), batch_size=16, shuffle=True, collate_fn=collate_fn)
test_bilstm_loader = DataLoader(BiLSTMDataset(test_word_ids, test_tags_filtered), batch_size=16, collate_fn=collate_fn)

# 5. BiLSTM-CRF ARCHITECTURE & TRAINING
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2id['<PAD>'])
        self.dropout = nn.Dropout(0.5) # Overfitting'i önlemek için eklendi
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.crf = CRF(tag_size, batch_first=True)
        
    def forward(self, sentences, tags=None):
        embeds = self.dropout(self.embedding(sentences))
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        
        mask = (sentences != word2id['<PAD>']).bool() 
        
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            pred_tags = self.crf.decode(emissions, mask=mask)
            return pred_tags

bilstm_model = BiLSTM_CRF(vocab_size, len(tag2id)).to(device)
bilstm_optimizer = AdamW(bilstm_model.parameters(), lr=2e-3)

print("\nTraining BiLSTM-CRF...")
bilstm_model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_bilstm_loader:
        sentences, labels = batch
        sentences = sentences.to(device)
        labels = labels.to(device)
        
        bilstm_optimizer.zero_grad()
        loss = bilstm_model(sentences, tags=labels)
        loss.backward()
        bilstm_optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_bilstm_loader):.3f}")

print("\nEvaluating BiLSTM-CRF...")
bilstm_model.eval()
bilstm_predictions, bilstm_true_labels = [], []

with torch.no_grad():
    for batch in test_bilstm_loader:
        sentences, labels = batch
        sentences = sentences.to(device)
        labels = labels.cpu().numpy()
        mask = (sentences.cpu().numpy() != word2id['<PAD>'])
        
        pred_tags_list = bilstm_model(sentences) 
        
        for i in range(len(pred_tags_list)):
            pred_seq = [id2tag[p] for p in pred_tags_list[i]]
            true_seq = [id2tag[l] for l, m in zip(labels[i], mask[i]) if m]
            
            bilstm_predictions.append(pred_seq)
            bilstm_true_labels.append(true_seq)

print("BiLSTM-CRF Classification Report:")
print(classification_report(bilstm_true_labels, bilstm_predictions))

# 6. COMPARISON & ANALYSIS
print("\n--- Comparison ---")
distil_f1 = f1_score(true_labels, predictions)
bilstm_f1 = f1_score(bilstm_true_labels, bilstm_predictions)
print(f"Transformer (DistilBERT) F1: {distil_f1:.4f}")
print(f"BiLSTM-CRF F1: {bilstm_f1:.4f}")