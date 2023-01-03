import torch
import torch.nn as nn


class LSTMStockPriceModel(nn.Module):
    def __init__(self):
        super(LSTMStockPriceModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=8,
            num_layers=2,
            batch_first=True,
        )

        self.linear = nn.Linear(8, 1)

    def forward(self, x, state=None):
        y, (h, c) = self.lstm(x, state)
        y = self.linear(y[:, -1, :])

        return y, h, c


if __name__ == '__main__':
    model = LSTMStockPriceModel()
    h = torch.zeros(2, 2, 8)
    c = torch.zeros(2, 2, 8)
    model_in = torch.rand(2, 30, 1)
    #print(model_in)
    #print(model_in.shape)
    for i in range(3):
        model_out, h, c = model(model_in, (h, c))
        model_in = torch.cat((model_in[:, 1:, :], model_out[:, :, None]), dim=1)
    #print(model_in)
    #print(model_in.shape)
    #print(model_out)
    #print(model_out.shape)
    #print(h_n)
    #print(h_n.shape)
    #print(c_n)
    #print(c_n.shape)
