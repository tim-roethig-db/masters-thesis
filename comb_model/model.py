import torch
import torch.nn as nn
from transformers import BertModel


class StockPriceModel(nn.Module):
    def __init__(self, n_news_features: int, rnn_n_layers: int, rnn_hidden_size: int):
        super(StockPriceModel, self).__init__()
        self.n_news_features = n_news_features
        if self.n_news_features > 0:
            self.bert = BertModel.from_pretrained('../models/finbert')
            for param in self.bert.parameters():
                param.requires_grad = False

            self.text_feature_ext = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, 128),
                nn.ReLU(),
                nn.Linear(128, n_news_features),
                nn.ReLU()
            )

        self.rnn = nn.GRU(
            input_size=n_news_features + 1,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_n_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(rnn_hidden_size, 1)

    def forward(self, x_price, x_news_input_ids, x_news_attention_mask, state=None):
        if self.n_news_features > 0:
            # apply news processing
            batch_size = x_news_input_ids.shape[0]
            # TODO find out whether to compute gradients for bert

            with torch.no_grad():
                x_news_input_ids = x_news_input_ids.flatten(start_dim=0, end_dim=1)
                x_news_attention_mask = x_news_attention_mask.flatten(start_dim=0, end_dim=1)
                last_hidden_state, pooler_output = self.bert(
                    input_ids=x_news_input_ids,
                    attention_mask=x_news_attention_mask,
                    return_dict=False
                )
                last_hidden_state = last_hidden_state.unflatten(0, (batch_size, int(last_hidden_state.shape[0] / batch_size)))

            # TODO use attention layer as feature extractor
            # TODO find out which part of last_hidden_state to use for output
            comp_feature_vect = self.text_feature_ext(last_hidden_state[:, :, 0, :])

            # cat price with news features
            x = torch.cat((x_price, comp_feature_vect), dim=2)
        else:
            x = x_price

        # run rnn
        y, state = self.rnn(x, state)

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
    news_attention_mask[:, seq_len - 1, :] = torch.randint(low=0, high=2, size=(batch_size, 512))

    x_price = torch.rand(batch_size, seq_len, 1)

    news_feature_vect = torch.zeros(size=(x_price.shape[0], x_price.shape[1], n_news_features))

    y, state = model(news_input_ids, news_attention_mask, x_price, news_feature_vect)
