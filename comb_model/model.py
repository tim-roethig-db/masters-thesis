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
                nn.Linear(768, n_news_features),
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
                last_hidden_state = last_hidden_state.unflatten(0, (
                batch_size, int(last_hidden_state.shape[0] / batch_size)))

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


class StockPriceModelARN(nn.Module):
    def __init__(self, n_news_features: int, seq_len: int):
        super(StockPriceModelARN, self).__init__()
        self.n_news_features = n_news_features

        self.activation = torch.nn.LeakyReLU()

        if self.n_news_features > 0:
            self.bert = BertModel.from_pretrained('../models/finbert')
            for param in self.bert.parameters():
                param.requires_grad = False
            """
            self.text_feature_ext = nn.Sequential(
                nn.Linear(768, 128),
                nn.Tanh(),
                nn.Linear(128, n_news_features),
                nn.Tanh()
            )
            """
            self.text_feature_ext = nn.Sequential(
                nn.Conv1d(768, 512, 4, padding=1),
                nn.Tanh(),
                nn.Conv1d(512, 256, 4, padding=1),
                nn.Tanh(),
            )
        """
        self.arn = nn.Sequential(
            nn.Linear(seq_len, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )
        
        self.arcn = nn.Sequential(
            nn.Conv1d(n_news_features+1, 32, 3, dilation=1),
            nn.Tanh(),
            nn.Conv1d(32, 64, 3, dilation=2),
            nn.Tanh(),
            nn.Conv1d(1, 1, 3, dilation=4),
            nn.Tanh(),
            nn.Conv1d(1, 1, 3, dilation=8),
            nn.Tanh(),
            nn.Linear(10, 1),
        )
        """
        self.price_feature_ext = nn.Sequential(
            nn.Linear(seq_len, 16),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(n_news_features + 16, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, x_price, x_news_input_ids, x_news_attention_mask, state=None):
        if self.n_news_features > 0:
            # news feature extaction
            batch_size = x_news_input_ids.shape[0]
            # TODO find out whether to compute gradients for bert

            with torch.no_grad():
                x_news_input_ids = x_news_input_ids.flatten(start_dim=0, end_dim=1)
                x_news_attention_mask = x_news_attention_mask.flatten(start_dim=0, end_dim=1)
                bert_out = self.bert(
                    input_ids=x_news_input_ids,
                    attention_mask=x_news_attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                print(bert_out.keys())
                print(bert_out['pooler_output'].shape)
                print(bert_out['last_hidden_state'].shape)

                print(bert_out['hidden_states'][0].shape)
                print(bert_out['hidden_states'][1].shape)
                print(bert_out['hidden_states'][2].shape)
                print(bert_out['hidden_states'][3].shape)
                print(len(bert_out['hidden_states']))
                last_hidden_state = bert_out['last_hidden_state'].unflatten(0, (
                batch_size, int(bert_out['last_hidden_state'].shape[0] / batch_size)))

            # TODO use attention layer as feature extractor
            # TODO find out which part of last_hidden_state to use for output
            print(last_hidden_state[:, :, 0, :].shape)
            cls_token = torch.permute(last_hidden_state[:, :, 0, :], (0, 2, 1))
            print(cls_token.shape)
            news_fv = self.text_feature_ext(cls_token)

            # price feature extraction
            print(x_price.shape)
            x_price = torch.permute(x_price, (0, 2, 1))
            price_fv = self.price_feature_ext(x_price)

            # cat price with news features
            print(news_fv.shape)
            print(price_fv.shape)
            x = torch.cat((price_fv, news_fv), dim=2)
        else:
            x_price = torch.permute(x_price, (0, 2, 1))
            x = self.price_feature_ext(x_price)

        y = self.reg_head(x)

        return y, state


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        v1 = torch.matmul(tau, self.w0) + self.b0
        v2 = self.f(torch.matmul(tau, self.w) + self.b)


        return torch.cat([v1, v2], dim=2)


class Time2Vec(nn.Module):
    def __init__(self, seq_len: int):
        super(Time2Vec, self).__init__()

        self.l1 = SineActivation(1, seq_len)
        self.fc1 = nn.Linear(seq_len, 2)

    def forward(self, x):
        x = self.l1(x)
        x = self.fc1(x)

        return x


class StockPriceModelTransformer(nn.Module):
    def __init__(self, n_news_features: int, seq_len: int):
        super(StockPriceModelTransformer, self).__init__()
        self.n_news_features = n_news_features

        if self.n_news_features > 0:
            self.bert = BertModel.from_pretrained('../models/finbert')
            for param in self.bert.parameters():
                param.requires_grad = False

            self.custom_pooler = nn.Sequential(
                nn.Linear(768, n_news_features),
                nn.Tanh()
            )

        self.t2v = Time2Vec(seq_len)

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
            transformer_in = torch.cat((x_price, pooler_output), dim=2)

        else:
            transformer_in = x_price

        print(transformer_in.shape)

        t2v_embedding = self.t2v(transformer_in)

        print(t2v_embedding.shape)

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
