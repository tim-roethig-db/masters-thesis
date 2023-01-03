import numpy as np
import pandas as pd
import torch

pd.options.display.max_columns = None


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.dtype = 'float32'

        df = df.pivot(index='company', columns='time_stamp', values='price')

        df = df.iloc[:, :40].dropna()

        self.x = df.iloc[:, :30].values.astype(self.dtype)
        self.x = self.x[:, :, None]
        print(self.x.shape)

        self.y = df.iloc[:, 31].values.astype(self.dtype)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    df = pd.read_csv('../data/stocks_prices_prep.csv', sep=';')

    ds = Dataset(df)
