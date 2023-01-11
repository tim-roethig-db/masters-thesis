import pandas as pd
import torch

from dataset import Dataset
from model import StockPriceModel


if __name__ == '__main__':
    batch_size = 2
    lr = 0.001
    epochs = 1

    # set device to cuda if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    print('Set up Data Loader...')
    price_df = pd.read_csv('../data/stocks_prices_prep.csv', sep=';', index_col=['company', 'time_stamp'])
    news_df = pd.read_csv('../data/articles_prep.csv', sep=';', index_col=['company', 'time_stamp'])
    train_set = Dataset(news_df, price_df, seq_len=30, test_len=5)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True)

    print('Loaded model to device...')
    model = StockPriceModel(n_news_features=32).float()
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

    print('Start train loop...')
    for epoch in range(1, epochs+1):
        epoch_loss = 0
        epoch_monitor_loss = 0

        # reset state every epoch
        state = None

        # iter over batches
        for x_news_input_ids, x_news_attention_mask, x_price, y in train_loader:
            # move data to device
            x_news_input_ids = x_news_input_ids.to(device)
            x_news_attention_mask = x_news_attention_mask.to(device)
            x_price = x_price.to(device)
            y = y[:, 0, :].to(device)

            # get prediction
            y_pred, state = model(x_news_input_ids, x_news_attention_mask, x_price)

            # compute loss
            batch_loss = loss(y_pred, y)
            epoch_loss += batch_loss
            batch_monitor_loss = monitor_loss(y_pred, y)
            epoch_monitor_loss += batch_monitor_loss

            # perform gradient step
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

            print(f'Batch MSELoss: {(batch_loss/batch_size):.5f} and MAELoss: {batch_monitor_loss/batch_size:.5f}')

        print(f'EPOCH: {epoch} of {epochs} with MSELoss: {epoch_loss/len(train_set):.5f} and MAELoss: {epoch_monitor_loss/len(train_set):.5f}')

    print('Save model...')
    torch.save(model.state_dict(), 'lstm.t7')
