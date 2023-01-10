import torch
import torch.nn as nn
from transformers import BertModel


class StockPriceModel(nn.Module):
    def __init__(self, n_news_features):
        super(StockPriceModel, self).__init__()

        self.n_news_features = n_news_features

        self.bert = BertModel.from_pretrained('../models/bert-base-uncased')

        self.text_feature_ext = nn.Sequential(
            nn.Linear(768, n_news_features),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=n_news_features + 1,
            hidden_size=n_news_features + 1,
            num_layers=2,
            batch_first=True,
        )

        self.linear = nn.Linear(n_news_features + 1, 1)

    def forward(self, news_input_ids, news_attention_mask, stock_price, state=None):
        # apply news processing for days with news
        # else fill with zeros
        news_feature_vect = torch.zeros(size=(stock_price.shape[0], stock_price.shape[1], self.n_news_features))
        for i in range(news_feature_vect.shape[1]):
            if news_input_ids[:, i, :].sum() > 0:
                last_hidden_state, pooler_output = self.bert(
                    input_ids=news_input_ids[:, i, :],
                    attention_mask=news_attention_mask[:, i, :],
                    return_dict=False
                )

                news_feature_vect[:, i, :] = self.text_feature_ext(pooler_output)

        # cat price with news features
        x = torch.cat((stock_price, news_feature_vect), dim=2)

        # run lstm
        y, state = self.lstm(x, state)
        y = self.linear(y[:, -1, :])

        return y, state


if __name__ == '__main__':
    batch_size = 16
    seq_len = 30
    n_news_features = 64

    model = StockPriceModel(n_news_features=n_news_features)

    news_input_ids = torch.zeros(batch_size, seq_len, 512, dtype=int)
    news_attention_mask = torch.zeros(batch_size, seq_len, 512, dtype=int)

    news_input_ids[:, seq_len - 1, :] = torch.randint(low=0, high=10000, size=(batch_size, 512))
    news_attention_mask[:, seq_len - 1, :] = torch.randint(low=0, high=1, size=(batch_size, 512))

    stock_price = torch.rand(batch_size, seq_len, 1)

    y, state = model(news_input_ids, news_attention_mask, stock_price)
