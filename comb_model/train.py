import pandas as pd
import torch
#from pynvml import *

from dataset import Dataset
from model import StockPriceModel


if __name__ == '__main__':
    print(torch.get_num_threads())
    batch_size = 1
    lr = 0.0001
    epochs = 2
    n_news_features = 16
    lstm_n_layers = 1
    lstm_hidden_size = 8

    # set device to cuda if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    print('Loaded model to device...')
    model = StockPriceModel(
        n_news_features=n_news_features,
        lstm_n_layers=lstm_n_layers,
        lstm_hidden_size=lstm_hidden_size
    ).float()
    model = torch.nn.DataParallel(model)
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
    mae_loss = torch.nn.L1Loss().to(device)

    print('Start training...')
    #price_df = pd.read_csv('../data/stocks_prices_prep.csv', sep=';', index_col=['company', 'time_stamp'])
    #news_df = pd.read_csv('../data/articles_prep.csv', sep=';', index_col=['company', 'time_stamp'])
    df = pd.read_csv('../data/dataset.csv', sep=';', index_col='time_stamp')

    #companys = sorted(list(set(price_df.index.get_level_values(0))))
    #for company in companys:
    #    if not price_df.loc[company].isnull().values.any():
    #print(f'Start training for {company}...')

    #print('Reset LSTM parameters...')
    #model.reset_lstm()

    print('Set up Data Loader...')
    #train_set = Dataset(
    #    company=company,
    #    news_df=news_df,
    #    price_df=price_df,
    #    seq_len=30,
    #    test_len=5
    #)
    train_set = Dataset(
        df=df,
        seq_len=30,
        test_len=5
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True)
    print(f'Series length: {len(train_set)}')

    print('Start train loop...')
    loss_df = list()
    for epoch in range(1, epochs+1):
        epoch_loss = 0
        epoch_monitor_loss = 0
        batch_monitor_loss = 0
        t_min = 0

        # reset state every epoch
        state = None

        # iter over batches
        for batch_idx, (x_news_input_ids, x_news_attention_mask, x_price, y, time_stamp) in enumerate(train_loader):
            """
            if torch.cuda.is_available():
                for i in range(4):
                    nvmlInit()
                    h = nvmlDeviceGetHandleByIndex(i)
                    info = nvmlDeviceGetMemoryInfo(h)
                    print(f'{i}_total: {info.total/1024**2}')
                    print(f'{i}_free : {info.free/1024**2}')
                    print(f'{i}_used : {info.used/1024**2}')
                print('------------------------------------------------')
            """
            # move data to device
            x_news_input_ids = x_news_input_ids.to(device)
            x_news_attention_mask = x_news_attention_mask.to(device)
            x_price = x_price.to(device)
            y = y.to(device)

            # get prediction
            y_pred, state = model(x_news_input_ids, x_news_attention_mask, x_price, state)

            state = [x.detach() for x in state]

            # compute loss
            batch_loss = loss(y_pred, y[:, 0, :])
            epoch_loss += batch_loss
            monitor_loss = mae_loss(y_pred, y[:, 0, :])
            epoch_monitor_loss += monitor_loss

            # perform gradient step
            model.zero_grad()
            batch_loss.backward(retain_graph=False)
            optimizer.step()

            #torch.cuda.empty_cache()

            p = 100
            if (batch_idx+1) % p == 0:
                batch_monitor_loss += monitor_loss
                print(f'{t_min} to {time_stamp.max()}: MAELoss: {batch_monitor_loss/p:.5f}')
                loss_df.append([epoch, batch_idx+1, (batch_monitor_loss/p).item()])

                batch_monitor_loss = 0
                t_min = time_stamp.min() + 1
            else:
                batch_monitor_loss += monitor_loss


        print(f'EPOCH: {epoch} of {epochs}: MSELoss: {epoch_loss/len(train_set):.5f}, MAELoss: {epoch_monitor_loss/len(train_set):.5f}')

    print('Save loss history...')
    loss_df = pd.DataFrame(
        columns=['epoch', 'iteration', 'MAE'],
        data=loss_df
    )
    loss_df.to_csv(f"loss_lr_{lr}_epoch_{epochs}.csv", index=False, sep=';')

    print('Save model...')
    torch.save(model.state_dict(), 'lstm.t7')
