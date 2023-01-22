import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

pd.options.display.max_columns = 10
pd.options.display.max_rows = 10
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
                max_length=515,
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


class Dataset_create(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, testing: bool, lag: int, seq_len: int, test_len: int):
        self.dtype = 'float32'
        self.seq_len = seq_len
        self.test_len = test_len
        self.tokenizer = BertTokenizer.from_pretrained('../models/finbert')
        self.bert = BertModel.from_pretrained('../models/finbert')
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False
        """
        if not testing:
            df = df.drop(df.tail(len(df)-3000).index)
        elif testing:
            df = df.drop(df.head(3000).index)

        df['title'] = df['title'].shift(-lag)
        df = df.drop(df.tail(lag).index)
        """
        tokens = df['title'].apply(lambda x: self.pd_tokenizer(x))
        input_ids = torch.stack([token['input_ids'][0] for token in tokens.values], dim=0)
        attention_mask = torch.stack([token['attention_mask'][0] for token in tokens.values], dim=0)
        with torch.no_grad():
            last_hidden_state, pooler_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False
            )
        df['title'] = [[x] for x in last_hidden_state]
        df = df.reset_index().reset_index()
        #df['alpha'] = abs(df['alpha'])
        self.x = df[['alpha', 'title', 'index']].values
        np.save('bert_dataset.npy', self.x)

    def __getitem__(self, idx: int):
        x_price = torch.from_numpy(self.x[idx:(idx+self.seq_len), 0].astype(self.dtype))
        x_price = x_price[:, None]

        x_news = self.x[idx:(idx+self.seq_len), 1]
        x_news = torch.stack([x[0] for x in x_news], dim=0)
        #x_news_input_ids = torch.stack([x['input_ids'][0] for x in x_news], dim=0)
        #x_news_attention_mask = torch.stack([x['attention_mask'][0] for x in x_news], dim=0)

        y = torch.from_numpy(self.x[(idx+self.seq_len):(idx+self.seq_len+self.test_len), 0].astype(self.dtype))
        y = y[:, None]

        time_stamp = self.x[idx:(idx+self.seq_len), 2].astype(int)

        #return x_news_input_ids, x_news_attention_mask, x_price, y, time_stamp
        return x_news, x_price, y, time_stamp

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
            """
            with torch.no_grad():
                x, pooler_output = self.bert(
                    input_ids=x['input_ids'],
                    attention_mask=x['attention_mask'],
                    return_dict=False
                )
            """
        else:
            x = {
                'input_ids': torch.zeros(1, 512, dtype=int),
                'attention_mask': torch.zeros(1, 512, dtype=int)
            }
            """
            with torch.no_grad():
                x, pooler_output = self.bert(
                    input_ids=x['input_ids'],
                    attention_mask=x['attention_mask'],
                    return_dict=False
                )
            """

        return x


class Dataset_test(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, testing: bool, lag: int, seq_len: int, test_len: int):
        self.dtype = 'float32'
        self.seq_len = seq_len
        self.test_len = test_len

        self.x = np.load('bert_dataset.npy', allow_pickle=True)
        print(self.x.shape)

        if not testing:
            self.x = self.x[:3000]
        elif testing:
            self.x = self.x[3000:]

        if lag > 0:
            tmp = self.x[lag:, 1]
            self.x = self.x[:-lag]
            self.x[:, 1] = tmp

    def __getitem__(self, idx: int):
        x_price = torch.from_numpy(self.x[idx:(idx+self.seq_len), 0].astype(self.dtype))
        x_price = x_price[:, None]

        x_news = self.x[idx:(idx+self.seq_len), 1]
        x_news = torch.stack([x[0] for x in x_news], dim=0)
        #x_news_input_ids = torch.stack([x['input_ids'][0] for x in x_news], dim=0)
        #x_news_attention_mask = torch.stack([x['attention_mask'][0] for x in x_news], dim=0)

        y = torch.from_numpy(self.x[(idx+self.seq_len):(idx+self.seq_len+self.test_len), 0].astype(self.dtype))
        y = y[:, None]

        time_stamp = self.x[idx:(idx+self.seq_len), 2].astype(int)

        #return x_news_input_ids, x_news_attention_mask, x_price, y, time_stamp
        return x_news, x_price, y, time_stamp

    def __len__(self):
        return len(self.x) - self.seq_len


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, testing: bool, lag: int, seq_len: int, test_len: int):
        self.dtype = 'float32'
        self.seq_len = seq_len
        self.test_len = test_len
        self.tokenizer = BertTokenizer.from_pretrained('../models/finbert')

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
        print(df.loc[676, 'title'])

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
    #price_df = pd.read_csv('../data/stocks_prices_prep.csv', sep=';', index_col=['company', 'time_stamp'])
    #news_df = pd.read_csv('../data/articles_prep.csv', sep=';', index_col=['company', 'time_stamp'])

    df = pd.read_csv('../data/dataset.csv', sep=';', index_col='time_stamp')

    #ds = Dataset_create(df, testing=False, lag=0, seq_len=40, test_len=5)
    ds = Dataset(df, testing=False, lag=1, seq_len=40, test_len=2)

    print(ds[0][4])
