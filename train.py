import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Обучить LSTM на AG News и оценить лучшие веса.")
parser.add_argument("--epochs", type=int, default=1, help="Число эпох; 0 — только оценка имеющегося lstm_model.pt")
args = parser.parse_args()
if args.epochs < 0:
    parser.error("--epochs должно быть неотрицательным")
if args.epochs == 0 and not Path("lstm_model.pt").exists():
    parser.error("Для --epochs 0 нужен lstm_model.pt. Сначала запустите обучение.")

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


train_data = pd.read_csv(
    'data/train.csv',
    header = None,
    names = ['label', 'title', 'description']
)

test_data = pd.read_csv(
    'data/test.csv',
    header = None,
    names = ['label', 'title', 'description']
)

train_counts = train_data['label'].value_counts()
test_counts = test_data['label'].value_counts()

train_data['text'] = train_data['title'] + ' ' + train_data['description']
test_data['text'] = test_data['title'] + ' ' + test_data['description']

train_data['label'] = train_data['label'] - 1
test_data['label'] = test_data['label'] - 1
train_data, val_data = train_test_split(
    train_data,
    test_size = 0.2,
    random_state = 42,
    stratify = train_data['label']
)

def tokenize(text):
    tokens = text.lower().split()
    return tokens

word_counts = Counter()
for news in train_data['text']:
    news = tokenize(news)
    word_counts.update(news)

vocab_size = 30000

stoi = {
    '<PAD>': 0,
    '<UNK>': 1
}
for word, count in word_counts.most_common(vocab_size - 2):
    stoi[word] = len(stoi)

def encode(text):
    tokens = tokenize(text)
    inputs = []

    for word in tokens:
        inputs.append(stoi.get(word, 1))
    return inputs

max_length = 64

def pad_sequence(input_ids, max_length):
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    if len(input_ids) < max_length:
        input_ids = input_ids + [0] * (max_length - len(input_ids))
    return input_ids

class NewsDataset(Dataset):
    def __init__(self, data, max_length):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row['text']
        targets = row['label']
        input_ids = encode(text)
        input_ids = pad_sequence(input_ids,self.max_length)
        input_ids = torch.tensor(input_ids, dtype = torch.long)
        targets = torch.tensor(targets, dtype = torch.long)
        return input_ids, targets

train_dataset = NewsDataset(train_data, max_length)
val_dataset = NewsDataset(val_data, max_length)
test_dataset = NewsDataset(test_data, max_length)

batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False)

embedding_dim = 128
hidden_size = 256
num_classes = 4

class LSTMModel(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_classes
        ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size,
            embedding_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_size,
            batch_first = True
        )
        self.classifier = nn.Linear(hidden_size,num_classes)

    def forward(self, input_ids):
        embedding = self.token_embedding(input_ids)
        output, (hidden, cell) = self.lstm(embedding)
        lengths = (input_ids != 0).sum(dim = 1)
        batch_indices = torch.arange(input_ids.shape[0],
            device = input_ids.device
        )
        last_indices = lengths - 1
        last_hidden = output[batch_indices, last_indices]
        logits = self.classifier(last_hidden)
        return logits

model = LSTMModel(vocab_size, embedding_dim, hidden_size, num_classes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)

best_val_loss = float('inf')
epochs = args.epochs
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_dataloader:
    # for input_ids, targets in train_dataloader: можно и так
        input_ids, targets = batch
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, targets)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 1 == 0:
        print('Epoch', epoch + 1,
            'Loss', loss.item())

    avg_loss = train_loss / len(train_dataloader)
    print('Avg_Loss', avg_loss)


    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, targets = batch
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            logits = model(input_ids)
            predictions = torch.argmax(logits, dim = 1)
            correct += (predictions == targets).sum().item()
            total += len(targets)
            loss = criterion(logits, targets)
            predictions = predictions.to('cpu')
            targets = targets.to('cpu')
            all_predictions.extend(predictions.tolist())
            all_targets.extend(targets.tolist())
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        print('Avg_Val_Loss', avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'lstm_model.pt')

    val_accuracy = (correct / total) * 100
    val_f1 = f1_score(all_targets, all_predictions, average = 'macro')
    print(val_accuracy)
    print('Macro F1', val_f1)

check_point = torch.load('lstm_model.pt',
    map_location = device,
    weights_only = True
)
model.load_state_dict(check_point)
model.eval()
test_loss = 0
correct = 0
total = 0
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, targets = batch
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        logits = model(input_ids)
        loss = criterion(logits, targets)
        predictions = torch.argmax(logits, dim = 1)
        correct += (predictions == targets).sum().item()
        total += len(targets)
        predictions = predictions.to('cpu')
        targets = targets.to('cpu')
        all_predictions.extend(predictions.tolist())
        all_targets.extend(targets.tolist())
        test_loss += loss.item() * len(targets)
    avg_test_loss = test_loss / total
    test_accuracy = (correct / total) * 100
    test_f1 = f1_score(all_targets, all_predictions, average = 'macro')
    print('Avg_Test_Loss', avg_test_loss)
    print('Test_Accuracy', test_accuracy)
    print('Test_f1', test_f1)

    check_point = {
        'model_state_dict': model.state_dict(),
        'stoi': stoi,
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_size': hidden_size,
        'num_classes': num_classes,
        'max_length': max_length
    }

    torch.save(check_point, 'lstm_checkpoint.pt')
