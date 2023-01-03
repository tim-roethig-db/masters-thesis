import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer

pd.options.display.max_columns = None


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.dtype = 'float32'
        self.tokenizer = BertTokenizer.from_pretrained('../models/bert-base-uncased')

        df = df.dropna(subset=['title', 'movement'])

        """
        self.x = [
            {
                key: value.astype('float32')
                for (key, value) in self.tokenizer(x, padding='max_length', max_length=512, truncation=True, return_tensors='np').items()
            }
            for x in df['title']
        ]
        """
        self.x = [self.tokenizer(x, padding='max_length', max_length=512, truncation=True, return_tensors='np') for x in df['title']]

        self.y = pd.get_dummies(df['movement']).values.astype(self.dtype)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    df = pd.read_csv('../data/tf_dataset.csv', sep=';')

    ds = Dataset(df)
