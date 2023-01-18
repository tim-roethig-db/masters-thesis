import torch
import torch.nn as nn
from transformers import BertModel


class StockPriceModel(nn.Module):
    def __init__(self, n_news_features: int, lstm_n_layers: int, lstm_hidden_size: int):
        super(StockPriceModel, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.n_news_features = n_news_features

        self.bert = BertModel.from_pretrained('../models/bert-base-uncased')

        self.text_feature_ext = nn.Sequential(
            nn.Linear(768, n_news_features),
            nn.Tanh()
        )

        self.lstm = nn.LSTM(
            input_size=n_news_features + 1,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_n_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(n_news_features + 1, 1)

    def forward(self, news_input_ids, news_attention_mask, stock_price, news_feature_vect, state=None):
        # apply news processing for days with news
        # else fill with zeros
        """
        for i in range(news_feature_vect.shape[1]):
            # if there is any input (>2 means more tokens than BOT and EOT)
            if news_input_ids[:, i, :].sum() > 0:
                last_hidden_state, pooler_output = self.bert(
                    input_ids=news_input_ids[:, i, :],
                    attention_mask=news_attention_mask[:, i, :],
                    return_dict=False
                )

                news_feature_vect[:, i, :] = self.text_feature_ext(pooler_output)
        """
        # cat price with news features
        #x = torch.cat((stock_price, news_feature_vect), dim=2)
        x = stock_price

        # run lstm
        y, state = self.lstm(x, state)
        y = self.linear(y[:, -1, :])

        return y, state

    def reset_lstm(self):
        self.lstm.reset_parameters()
        self.linear.reset_parameters()


if __name__ == '__main__':
    batch_size = 16
    seq_len = 30
    n_news_features = 64

    model = StockPriceModel(n_news_features=n_news_features)

    news_input_ids = torch.zeros(batch_size, seq_len, 512, dtype=int)
    news_attention_mask = torch.zeros(batch_size, seq_len, 512, dtype=int)

    news_input_ids[:, seq_len - 1, :] = torch.randint(low=0, high=10000, size=(batch_size, 512))
    news_attention_mask[:, seq_len - 1, :] = torch.randint(low=0, high=2, size=(batch_size, 512))

    x_price = torch.rand(batch_size, seq_len, 1)

    news_feature_vect = torch.zeros(size=(x_price.shape[0], x_price.shape[1], n_news_features))

    y, state = model(news_input_ids, news_attention_mask, x_price, news_feature_vect)
