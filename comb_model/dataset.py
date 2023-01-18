import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer

pd.options.display.max_columns = 10
pd.options.display.max_rows = None
#pd.options.expand_frame_repr = False
pd.set_option('expand_frame_repr', False)


class Dataset_old(torch.utils.data.Dataset):
    def __init__(self, company: str, news_df: pd.DataFrame, price_df: pd.DataFrame, seq_len: int = 30, test_len: int = 5):
        self.dtype = 'float32'
        self.seq_len = seq_len
        self.test_len = test_len
        self.tokenizer = BertTokenizer.from_pretrained('../models/bert-base-uncased')
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        price_df = price_df.loc[company]

        news_df = news_df.loc[company]
        news_df = news_df['title'].dropna()
        news_df = news_df.groupby(['time_stamp']).apply(lambda x: ' '.join(x.values))

        df = price_df.join(news_df)

        # TODO drop complete company if it has price na
        df = df.dropna(subset=['price'])
        df['title'] = df['title'].apply(lambda x: self.pd_tokenizer(x))

        df = df.reset_index().reset_index()

        self.x = df[['price', 'title', 'index']].values

    def __getitem__(self, idx: int):
        x_price = torch.from_numpy(self.x[idx:(idx+self.seq_len), 0].astype(self.dtype))
        x_price = x_price[:, None]

        x_news = self.x[idx:(idx+self.seq_len), 1]
        x_news_input_ids = torch.stack([x['input_ids'][0] for x in x_news], dim=0)
        x_news_attention_mask = torch.stack([x['attention_mask'][0] for x in x_news], dim=0)

        y = torch.from_numpy(self.x[(idx+self.seq_len):(idx+self.seq_len+self.test_len), 0].astype(self.dtype))
        y = y[:, None]

        time_stamp = self.x[idx:(idx+self.seq_len), 2].astype(int)

        return x_news_input_ids, x_news_attention_mask, x_price, y, time_stamp

    def __len__(self):
        return len(self.x) - self.seq_len

    def pd_tokenizer(self, x):
        if x is not np.nan:
            x = self.tokenizer(
                x,
                padding='max_length',
                max_length=256,
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=False
            )

        else:
            x = {
                'input_ids': torch.zeros(1, 512, dtype=int),
                'attention_mask': torch.zeros(1, 512, dtype=int)
            }

        return x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 30, test_len: int = 5):
        self.dtype = 'float32'
        self.seq_len = seq_len
        self.test_len = test_len
        self.tokenizer = BertTokenizer.from_pretrained('../models/bert-base-uncased')

        df['title'] = df['title'].apply(lambda x: self.pd_tokenizer(x))

        df = df.reset_index().reset_index()

        self.x = df[['alpha', 'title', 'index']].values

    def __getitem__(self, idx: int):
        x_price = torch.from_numpy(self.x[idx:(idx+self.seq_len), 0].astype(self.dtype))
        x_price = x_price[:, None]

        x_news = self.x[idx:(idx+self.seq_len), 1]
        x_news_input_ids = torch.stack([x['input_ids'][0] for x in x_news], dim=0)
        x_news_attention_mask = torch.stack([x['attention_mask'][0] for x in x_news], dim=0)

        y = torch.from_numpy(self.x[(idx+self.seq_len):(idx+self.seq_len+self.test_len), 0].astype(self.dtype))
        y = y[:, None]

        time_stamp = self.x[idx:(idx+self.seq_len), 2].astype(int)

        return x_news_input_ids, x_news_attention_mask, x_price, y, time_stamp

    def __len__(self):
        return len(self.x) - self.seq_len

    def pd_tokenizer(self, x):
        if x is not np.nan:
            x = self.tokenizer(
                x,
                padding='max_length',
                max_length=512,
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=False
            )

        else:
            x = {
                'input_ids': torch.zeros(1, 512, dtype=int),
                'attention_mask': torch.zeros(1, 512, dtype=int)
            }

        return x


if __name__ == '__main__':
    #price_df = pd.read_csv('../data/stocks_prices_prep.csv', sep=';', index_col=['company', 'time_stamp'])
    #news_df = pd.read_csv('../data/articles_prep.csv', sep=';', index_col=['company', 'time_stamp'])

    df = pd.read_csv('../data/dataset.csv', sep=';', index_col='time_stamp')

    ds = Dataset(df)

    print(ds[0])
