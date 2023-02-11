import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

pd.options.display.max_columns = 10
pd.options.display.max_rows = 10
pd.set_option('expand_frame_repr', False)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, news_df: pd.DataFrame, price_df: pd.DataFrame, testing: bool, lag: int, seq_len: int, test_len: int):
        self.dtype = 'float32'
        self.seq_len = seq_len
        self.test_len = test_len

        #tokenizer = BertTokenizer.from_pretrained('../models/finbert')
        tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

        news_df = news_df[~news_df['topic'].isin([-1])]
        news_df = news_df.groupby('time_stamp')['title'].apply(lambda x: ' '.join(x.values))

        news_df.index = pd.to_datetime(news_df.index)
        price_df.index = pd.to_datetime(price_df.index)

        df = pd.merge_asof(
            price_df, news_df,
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta('5d'),
            direction='backward'
        )

        df['title'] = df['title'].fillna('')

        df['title'] = df['title'].shift(-lag)
        df = df.drop(df.tail(lag).index)

        if not testing:
            df = df.drop(df.tail(len(df)-3000).index)
        elif testing:
            df = df.drop(df.head(3000).index)

        encoding = tokenizer(
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

        #df['alpha'] = df['alpha'].ewm(halflife=3).mean()
        #print(df)
        #df['alpha'] = abs(df['alpha'])

        df['up'] = 0
        df.loc[df['alpha'] > 0, 'up'] = 1

        df = df.reset_index().reset_index()
        df = df[['index', 'alpha', 'input_ids', 'attention_mask', 'up']]

        self.x = df.values

    def __getitem__(self, idx: int):
        time_stamp = self.x[idx:(idx + self.seq_len), 0].astype(int)

        x_price = torch.from_numpy(self.x[idx:(idx+self.seq_len), 1].astype(self.dtype))
        x_price = x_price[:, None]

        x_news_input_ids = torch.stack(
            [x[0] for x in self.x[idx:(idx+self.seq_len), 2]],
            dim=0
        )
        x_news_attention_mask = torch.stack(
            [x[0] for x in self.x[idx:(idx+self.seq_len), 3]],
            dim=0
        )

        #y = torch.from_numpy(self.x[(idx+self.seq_len):(idx+self.seq_len+self.test_len), 1].astype(self.dtype))
        y = torch.from_numpy(np.array([self.x[(idx+self.seq_len), 4]]).astype(self.dtype))

        return time_stamp, x_price, x_news_input_ids, x_news_attention_mask, y

    def __len__(self):
        return len(self.x) - self.seq_len


if __name__ == '__main__':
    news_df = pd.read_csv('../data/rwe_news_dataset.csv', sep=';')
    price_df = pd.read_csv('../data/rwe_price_dataset.csv', sep=';', index_col='time_stamp')

    ds = Dataset(news_df, price_df, testing=False, lag=1, seq_len=5, test_len=2)

    print(ds[0])
