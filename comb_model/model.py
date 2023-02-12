import torch
import torch.nn as nn
from transformers import BertModel


class StockPriceModel(nn.Module):
    def news_processing(self, x_news_input_ids, x_news_attention_mask):
        batch_size = x_news_input_ids.shape[0]

        with torch.no_grad():
            x_news_input_ids = x_news_input_ids.flatten(start_dim=0, end_dim=1)
            x_news_attention_mask = x_news_attention_mask.flatten(start_dim=0, end_dim=1)
            bert_out = self.bert(
                input_ids=x_news_input_ids,
                attention_mask=x_news_attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            last_hidden_state = bert_out['last_hidden_state'].unflatten(
                0, (batch_size, int(bert_out['last_hidden_state'].shape[0] / batch_size))
            )
            second_last_hidden_state = bert_out['hidden_states'][-2].unflatten(
                0, (batch_size, int(bert_out['last_hidden_state'].shape[0] / batch_size))
            )

        # pooled = last_hidden_state[:, :, 0, :]
        pooled = second_last_hidden_state.mean(dim=2)
        news_fv = self.text_feature_ext(pooled)

        return news_fv


class StockPriceModelRNN(StockPriceModel):
    def __init__(self, n_news_features: int, rnn_n_layers: int, rnn_hidden_size: int):
        super(StockPriceModelRNN, self).__init__()
        self.model_name = 'model_rnn'
        self.n_news_features = n_news_features
        if self.n_news_features > 0:
            #self.bert = BertModel.from_pretrained('../models/finbert')
            self.bert = BertModel.from_pretrained('ProsusAI/finbert')
            for param in self.bert.parameters():
                param.requires_grad = False

            self.text_feature_ext = nn.Sequential(
                nn.Linear(768, n_news_features),
                nn.ReLU()
            )

        self.rnn = nn.GRU(
            input_size=n_news_features + 1,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_n_layers,
            batch_first=True,
        )

        self.clf_head = nn.Sequential(
            nn.Linear(rnn_hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x_price, x_news_input_ids, x_news_attention_mask, state=None):
        if self.n_news_features > 0:
            # news feature extraction
            news_fv = self.news_processing(x_news_input_ids, x_news_attention_mask)

            # cat price with news features
            y = torch.cat((x_price, news_fv), dim=2)
        else:
            y = x_price

        # run rnn
        y, state = self.rnn(y, state)

        y = self.clf_head(y[:, -1, :])

        return y, state


class StockPriceModelARN(StockPriceModel):
    def __init__(self, n_news_features: int, seq_len: int):
        super(StockPriceModelARN, self).__init__()
        self.model_name = 'model_arn'
        self.n_news_features = n_news_features

        if self.n_news_features > 0:
            #self.bert = BertModel.from_pretrained('../models/finbert')
            self.bert = BertModel.from_pretrained('ProsusAI/finbert')
            for param in self.bert.parameters():
                param.requires_grad = False

            self.text_feature_ext = nn.Sequential(
                nn.Linear(768, n_news_features),
                nn.Tanh()
            )

        self.reg_head = nn.Sequential(
            nn.Linear((n_news_features+1)*seq_len, 32),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(32, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x_price, x_news_input_ids, x_news_attention_mask, state=None):
        if self.n_news_features > 0:
            # news feature extraction
            news_fv = self.news_processing(x_news_input_ids, x_news_attention_mask)

            # cat price with news features
            y = torch.cat((x_price, news_fv), dim=2)
            y = torch.flatten(y, start_dim=1, end_dim=2)
        else:
            y = x_price[:, :, 0]

        y = self.reg_head(y)

        return y, state


class Time2Vec(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(Time2Vec, self).__init__()

        self.out_features = out_features
        self.w_linear = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b_linear = nn.parameter.Parameter(torch.randn(1))
        self.w_periodic = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b_periodic = nn.parameter.Parameter(torch.randn(out_features - 1))

    def forward(self, tau):
        linear = torch.matmul(tau, self.w_linear) + self.b_linear
        periodic = torch.sin(torch.matmul(tau, self.w_periodic) + self.b_periodic)

        t2v = torch.cat([linear, periodic], dim=2)

        return t2v


class StockPriceModelTransformer(nn.Module):
    def __init__(self, n_news_features: int, seq_len: int):
        super(StockPriceModelTransformer, self).__init__()
        self.model_name = 'model_tf'
        self.n_news_features = n_news_features

        if self.n_news_features > 0:
            self.bert = BertModel.from_pretrained('../models/finbert')
            for param in self.bert.parameters():
                param.requires_grad = False

            self.custom_pooler = nn.Sequential(
                nn.Linear(768, n_news_features),
                nn.Tanh()
            )

        self.t2v = Time2Vec(n_news_features+1, 2*(n_news_features+1))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=3*(1+n_news_features),
            nhead=3,
            dim_feedforward=64,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,
        )

        self.global_avg_pooling = nn.AvgPool1d(3*(1+n_news_features))

        self.reg_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(seq_len, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x_price, x_news_input_ids, x_news_attention_mask, state=None):
        if self.n_news_features > 0:
            # compute bert embedding
            batch_size = x_news_input_ids.shape[0]
            with torch.no_grad():
                x_news_input_ids = x_news_input_ids.flatten(start_dim=0, end_dim=1)
                x_news_attention_mask = x_news_attention_mask.flatten(start_dim=0, end_dim=1)
                bert_out = self.bert(
                    input_ids=x_news_input_ids,
                    attention_mask=x_news_attention_mask,
                    return_dict=True
                )

            last_hidden_state = bert_out['last_hidden_state'].unflatten(
                0, (batch_size, int(bert_out['last_hidden_state'].shape[0] / batch_size))
            )
            cls = last_hidden_state[:, :, 0, :]
            pooler_output = self.custom_pooler(cls)
            x = torch.cat((x_price, pooler_output), dim=2)

        else:
            x = x_price

        t2v_embedding = self.t2v(x)

        transformer_in = torch.cat((t2v_embedding, x), dim=2)

        transformer_out = self.encoder(transformer_in)

        x = self.global_avg_pooling(transformer_out)
        x = x[:, :, 0]

        y = self.reg_head(x)

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
