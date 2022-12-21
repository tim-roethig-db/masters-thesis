import pandas as pd
import torch
from transformers import BertTokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        print(df)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        #self.labels = [labels[label] for label in df['category']]
        #self.texts = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for text in df['text']]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    df = pd.read_csv('../data/articles_prep.csv', sep=';')
