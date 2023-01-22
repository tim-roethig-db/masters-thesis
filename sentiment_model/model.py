import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class StockPriceModel(nn.Module):
    def __init__(self, news_features: bool, rnn_n_layers: int, rnn_hidden_size: int):
        super(StockPriceModel, self).__init__()
        self.news_features = news_features
        if self.news_features:
            self.sentiment_bert = AutoModelForSequenceClassification.from_pretrained('../models/finbert')
            for param in self.sentiment_bert.parameters():
                param.requires_grad = False

        self.sequence_model = nn.GRU(
            input_size=3 + 1,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_n_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(rnn_hidden_size, 1)

    def forward(self, x_price, x_news_input_ids, x_news_attention_mask, state=None):
        if self.news_features:
            # apply news processing
            batch_size = x_news_input_ids.shape[0]

            with torch.no_grad():
                x_news_input_ids = x_news_input_ids.flatten(start_dim=0, end_dim=1)
                x_news_attention_mask = x_news_attention_mask.flatten(start_dim=0, end_dim=1)
                bert_out = self.sentiment_bert(
                    input_ids=x_news_input_ids,
                    attention_mask=x_news_attention_mask,
                    return_dict=True
                )
                sentiment = bert_out['logits'].unflatten(0, (batch_size, int(bert_out['logits'].shape[0] / batch_size)))
            # cat price with news features
            x = torch.cat((x_price, sentiment), dim=2)
        else:
            x = x_price

        # run rnn
        y, state = self.sequence_model(x, state)

        y = self.linear(y[:, -1, :])

        return y, state
