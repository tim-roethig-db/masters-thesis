import pandas as pd
import torch

from dataset import Dataset
from model import LSTMStockPriceModel


if __name__ == '__main__':
    batch_size = 16
    lr = 0.0001
    epochs = 50

    # set device to cuda if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('Set up Data Loader...')
    df = pd.read_csv('../data/stocks_prices_prep.csv', sep=';')#.sample(frac=1).head(4)
    train_set = Dataset(df)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True)

    print('Loaded model to device...')
    model = LSTMStockPriceModel().float()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable model parameters: {total_params}')

    print('Setup Adam optimizer...')
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    print('Setup loss function...')
    loss = torch.nn.MSELoss(reduction='sum').to(device)
    monitor_loss = torch.nn.L1Loss()

    # train loop
    for epoch in range(1, epochs+1):
        epoch_loss = 0
        epoch_monitor_loss = 0

        # iter over batches
        for x, y in train_loader:
            # move data to device
            x = x.to(device)
            y = y.to(device)

            # get prediction
            y_pred, _, _ = model(x)

            # compute loss
            batch_loss = loss(y_pred, y)
            epoch_loss += batch_loss
            batch_monitor_loss = monitor_loss(y_pred, y)
            epoch_monitor_loss += batch_monitor_loss

            # perform gradient step
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        print(f'EPOCH: {epoch} of {epochs} with MSELoss: {epoch_loss/len(train_set):.5f} and MAELoss: {epoch_monitor_loss/len(train_set):.5f}')

    print('Save model...')
    torch.save(model.state_dict(), 'lstm.t7')
