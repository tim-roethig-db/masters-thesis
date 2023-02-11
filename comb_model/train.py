import os
from datetime import datetime
import pandas as pd
import torch
from pynvml import *

from dataset import Dataset
from model import StockPriceModelRNN, StockPriceModelARN, StockPriceModelTransformer


if __name__ == '__main__':
    batch_size = 8
    lr = 0.001
    epochs = 1
    n_news_features = 16
    rnn_n_layers = 1
    rnn_hidden_size = 16
    seq_len = 10
    lag = 0

    # set device to cuda if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    print('Loaded model to device...')
    """
    model = StockPriceModelRNN(
        n_news_features=n_news_features,
        rnn_n_layers=rnn_n_layers,
        rnn_hidden_size=rnn_hidden_size
    ).float()
    """
    model = StockPriceModelARN(
        n_news_features=n_news_features,
        seq_len=seq_len
    ).float()
    """
    model = StockPriceModelTransformer(
        n_news_features=n_news_features,
        seq_len=seq_len
    ).float()
    """
    #model = torch.nn.DataParallel(model)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable model parameters: {total_params}')

    print('Setup Adam optimizer...')
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    print('Setup loss function...')
    #loss = torch.nn.MSELoss().to(device)
    loss = torch.nn.BCELoss().to(device)
    mae_loss = torch.nn.L1Loss(reduction='sum').to(device)

    print('Start training...')
    #news_df = pd.read_csv('../data/rwe_news_dataset.csv', sep=';')
    #price_df = pd.read_csv('../data/rwe_price_dataset.csv', sep=';', index_col='time_stamp')
    news_df = pd.read_csv('data/rwe_news_dataset.csv', sep=';')
    price_df = pd.read_csv('data/rwe_price_dataset.csv', sep=';', index_col='time_stamp')
    print('Set up Data Loader...')
    train_set = Dataset(
        news_df=news_df,
        price_df=price_df,
        testing=False,
        lag=lag,
        seq_len=seq_len,
        test_len=1,
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
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
        for batch_idx, (time_stamp, x_price, x_news_input_ids, x_news_attention_mask, y) in enumerate(train_loader):
            #print(batch_idx)

            if torch.cuda.is_available():
                for i in range(4):
                    nvmlInit()
                    h = nvmlDeviceGetHandleByIndex(i)
                    info = nvmlDeviceGetMemoryInfo(h)
                    print(f'{i}_total: {info.total/1024**2}')
                    print(f'{i}_free : {info.free/1024**2}')
                    print(f'{i}_used : {info.used/1024**2}')
                print('------------------------------------------------')

            # move data to device
            x_price = x_price.to(device)
            x_news_input_ids = x_news_input_ids.to(device)
            x_news_attention_mask = x_news_attention_mask.to(device)
            y = y.to(device)

            # get prediction
            y_pred, state = model(x_price, x_news_input_ids, x_news_attention_mask, state)
            #state = state.detach()
            #state = [x.detach() for x in state]
            #y_pred = torch.ones((batch_size, 1))
            #state = None

            # compute loss
            #y = y[:, :, 0]
            y_pred_class = (y_pred > 0.5).float()
            batch_loss = loss(y_pred, y)
            epoch_loss += batch_loss
            #monitor_loss = mae_loss(y_pred, y)
            monitor_loss = torch.sum(y_pred_class == y)
            epoch_monitor_loss += monitor_loss

            # perform gradient step
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

            p = 100 // batch_size
            if (batch_idx+1) % p == 0:
                batch_monitor_loss += monitor_loss
                print(f'{t_min} to {time_stamp.max()}: MAELoss: {batch_monitor_loss/(p*batch_size):.5f}')
                loss_df.append([epoch, batch_idx+1, (batch_monitor_loss/(p*batch_size)).item()])

                batch_monitor_loss = 0
                t_min = time_stamp.min() + batch_size
            else:
                batch_monitor_loss += monitor_loss
        print(epoch_loss)
        print(f'EPOCH: {epoch} of {epochs}: MSELoss: {epoch_loss/len(train_set):.5f}, MAELoss: {epoch_monitor_loss/len(train_set):.5f}')

    print('Save model...')
    file_name = f'{model.model_name}_{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}'
    os.system(f'mkdir {file_name}')

    loss_df = pd.DataFrame(
        columns=['epoch', 'iteration', 'MAE'],
        data=loss_df
    )
    loss_df.to_csv(f'{file_name}/train_loss.csv', index=False, sep=';')

    torch.save(model.state_dict(), f'{file_name}/model.t7')
    pd.DataFrame({
        'batch_size': [batch_size],
        'lr': lr,
        'epochs': epochs,
        'n_news_features': n_news_features,
        'rnn_n_layers': rnn_n_layers,
        'rnn_hidden_size': rnn_hidden_size,
        'seq_len': seq_len,
        'lag': lag
    }).to_json(f'{file_name}/conf.json')

    os.system(f'zip -r {file_name}.zip {file_name}')


