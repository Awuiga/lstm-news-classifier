"""Оценка доверенного LSTM checkpoint на validation и официальном test AG News."""
import argparse
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# Эквивалент пользовательского класса, без запуска train.py или predict.py.
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        output, _ = self.lstm(self.token_embedding(input_ids))
        lengths = (input_ids != 0).sum(dim=1)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        return self.classifier(output[rows, lengths - 1])


def read_data(path):
    df = pd.read_csv(path, header=None, names=['label', 'title', 'description'])
    if df.isna().any().any() or not df['label'].isin([1, 2, 3, 4]).all():
        raise ValueError('Ожидается полный AG News CSV без заголовка, метки 1–4.')
    df['text'] = df['title'] + ' ' + df['description']
    df['label'] -= 1
    return df


class EncodedNews(Dataset):
    def __init__(self, df, stoi, max_length):
        self.texts = df['text'].tolist()
        self.targets = df['label'].tolist()
        self.stoi, self.max_length = stoi, max_length

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        ids = [self.stoi.get(word, 1) for word in self.texts[index].lower().split()][:self.max_length]
        if not ids:
            raise ValueError('Пустые тексты не поддерживаются исходной моделью.')
        ids += [0] * (self.max_length - len(ids))
        return torch.tensor(ids), self.targets[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='lstm_checkpoint.pt')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output', default='reports/metrics.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    data_root = Path(args.data_root)
    full_train, test = read_data(data_root/'train.csv'), read_data(data_root/'test.csv')
    train, val = train_test_split(full_train, test_size=.2, random_state=42, stratify=full_train['label'])
    counts = Counter()
    for text in train['text']:
        counts.update(text.lower().split())
    expected_stoi = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counts.most_common(cp['vocab_size'] - 2):
        expected_stoi[word] = len(expected_stoi)
    if cp['stoi'] != expected_stoi:
        raise ValueError('Словарь checkpoint не совпал с восстановленным train.')
    model = LSTMModel(*(cp[k] for k in ['vocab_size', 'embedding_dim', 'hidden_size', 'num_classes'])).to(device)
    model.load_state_dict(cp['model_state_dict'], strict=True)
    model.eval()
    labels = ['World', 'Sports', 'Business', 'Sci/Tech']
    def measure(df):
        actual, predicted, loss_sum = [], [], 0.
        dataset = EncodedNews(df, cp['stoi'], cp['max_length'])
        with torch.no_grad():
            for x, y in DataLoader(dataset, batch_size=64, shuffle=False):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
                actual.extend(y.cpu().tolist())
                predicted.extend(logits.argmax(dim=1).cpu().tolist())
        return {'samples': len(df), 'accuracy': accuracy_score(actual, predicted),
                'cross_entropy_per_news': loss_sum / len(df),
                'classification_report': classification_report(actual, predicted, labels=range(4),
                                          target_names=labels, output_dict=True, zero_division=0),
                'confusion_matrix': confusion_matrix(actual, predicted, labels=range(4)).tolist()}
    report = {
        'evaluation_date': datetime.now(timezone(timedelta(hours=3))).date().isoformat(),
        'classes': labels, 'train_samples': len(train), 'validation': measure(val), 'test': measure(test),
        'parameters': sum(p.numel() for p in model.parameters()),
        'configuration': {k: cp[k] for k in ['vocab_size', 'embedding_dim', 'hidden_size', 'num_classes', 'max_length']},
        'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'normalized_csv_sha256': {name: hashlib.sha256((data_root/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                  for name in ['train.csv', 'test.csv']},
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'device': device},
        'notes': ['Existing complete checkpoint evaluated without training or tuning.',
                  'Validation reconstructed with stratified train_test_split, random_state 42.',
                  'Entire train vocabulary exactly matched saved stoi.',
                  'Checkpoint has no dataset hash, training indices or epoch metadata.',
                  'No guarantee that test had never been inspected in earlier development.']
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for split in ['validation', 'test']:
        result = report[split]
        print(split, 'samples', result['samples'], 'accuracy', result['accuracy'],
              'macro F1', result['classification_report']['macro avg']['f1-score'],
              'loss', result['cross_entropy_per_news'], flush=True)
    print('parameters', report['parameters'])


if __name__ == '__main__':
    main()
