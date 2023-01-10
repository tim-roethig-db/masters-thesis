import pandas as pd
import torch

pd.options.display.max_columns = None


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 30, test_len: int = 5):
        self.dtype = 'float32'
        self.seq_len = seq_len
        self.test_len = test_len

        df = df.pivot(index='company', columns='time_stamp', values='price')

        #df = df.iloc[:, :self.seq_len+self.test_len].dropna()
        df = df.loc['salzgitter ag'].dropna()

        #self.x = df.iloc[:, :self.seq_len].values.astype(self.dtype)
        #self.x = torch.from_numpy(self.x[:, :, None])

        #self.y = df.iloc[:, self.seq_len:self.seq_len+self.test_len].values.astype(self.dtype)
        #self.y = torch.from_numpy(self.y[:, :, None])

        self.x = df.values.astype(self.dtype)
        self.x = torch.from_numpy(self.x[:, None])

    def __getitem__(self, idx: int):
        #return self.x[idx], self.y[idx]
        return self.x[idx:(idx+self.seq_len)], self.x[(idx+self.seq_len):(idx+self.seq_len+self.test_len)]

    def __len__(self):
        return len(self.x) - self.seq_len


if __name__ == '__main__':
    df = pd.read_csv('../data/stocks_prices_prep.csv', sep=';')

    ds = Dataset(df)
    print(len(ds))
    print(ds[0])
