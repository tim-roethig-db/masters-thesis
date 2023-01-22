import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, testing: bool, lag: int, seq_len: int, test_len: int):
        self.dtype = 'float32'
        self.seq_len = seq_len
        self.test_len = test_len
        self.tokenizer = AutoTokenizer.from_pretrained('../models/finbert')

        df['title'] = df['title'].fillna('')

        encoding = self.tokenizer(
            df['title'].tolist(),
            padding='longest',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False,
            return_length=True
        )

        df['input_ids'] = [[x] for x in encoding['input_ids']]
        df['attention_mask'] = [[x] for x in encoding['attention_mask']]
        df['length'] = [x.item() for x in encoding['length']]

        #df['alpha'] = abs(df['alpha'])

        df = df.reset_index().reset_index()
        df = df[['index', 'title', 'alpha', 'input_ids', 'attention_mask', 'length']]

        if not testing:
            df = df.drop(df.tail(len(df)-3000).index)
        elif testing:
            df = df.drop(df.head(3000).index)

        df['title'] = df['title'].shift(-lag)
        df = df.drop(df.tail(lag).index)

        self.x = df.values

    def __getitem__(self, idx: int):
        time_stamp = self.x[idx:(idx + self.seq_len), 0].astype(int)

        x_price = torch.from_numpy(self.x[idx:(idx+self.seq_len), 2].astype(self.dtype))
        x_price = x_price[:, None]

        x_news_input_ids = torch.stack(
            [x[0] for x in self.x[idx:(idx+self.seq_len), 3]],
            dim=0
        )
        x_news_attention_mask = torch.stack(
            [x[0] for x in self.x[idx:(idx+self.seq_len), 4]],
            dim=0
        )

        y = torch.from_numpy(self.x[(idx+self.seq_len):(idx+self.seq_len+self.test_len), 2].astype(self.dtype))
        y = y[:, None]

        return time_stamp, x_price, x_news_input_ids, x_news_attention_mask, y

    def __len__(self):
        return len(self.x) - self.seq_len


if __name__ == '__main__':
    df = pd.read_csv('../data/dataset.csv', sep=';', index_col='time_stamp')

    #ds = Dataset_create(df, testing=False, lag=0, seq_len=40, test_len=5)
    ds = Dataset(df, testing=False, lag=1, seq_len=40, test_len=2)

    print(ds[0])
