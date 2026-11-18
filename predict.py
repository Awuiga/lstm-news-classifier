import torch
import torch.nn as nn

def tokenize(text):
    tokens = text.lower().split()
    return tokens

def encode(text):
    tokens = tokenize(text)
    inputs = []

    for word in tokens:
        inputs.append(stoi.get(word, 1))
    return inputs

def pad_sequence(input_ids, max_length):
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    if len(input_ids) < max_length:
        input_ids = input_ids + [0] * (max_length - len(input_ids))
    return input_ids

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

check_point =torch.load('lstm_checkpoint.pt',
    map_location = device,
    weights_only = True
)

model = LSTMModel(
    check_point['vocab_size'],
    check_point['embedding_dim'],
    check_point['hidden_size'],
    check_point['num_classes'],
).to(device)

model.load_state_dict(check_point['model_state_dict'])
model.eval()

class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
text = "Manchester United won the football match against Chelsea"

stoi = check_point['stoi']
max_length = check_point['max_length']

input_ids = encode(text)
input_ids = pad_sequence(input_ids, max_length)
input_ids = torch.tensor(input_ids, dtype = torch.long)
input_ids = input_ids.unsqueeze(0)
input_ids = input_ids.to(device)

with torch.no_grad():
    logits = model(input_ids)
    probabilities = torch.softmax(logits, dim = 1)
    predictions = torch.argmax(logits, dim = 1).item()
    class_name = class_names[predictions]
    confidence = probabilities[0, predictions].item() * 100

print('Text:', text)
print('Category:', class_name)
print('Confidence:', confidence)
