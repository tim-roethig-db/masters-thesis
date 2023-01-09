import pandas as pd
import torch

from dataset import Dataset
from model import BERTNewsClf


if __name__ == '__main__':
    batch_size = 2
    lr = 0.01
    epochs = 3

    print(f'CUDA available: {torch.cuda.is_available()}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('Set up Data Loader...')
    df = pd.read_csv('../data/tf_dataset.csv', sep=';')#.sample(frac=1).head(40)
    train_set = Dataset(df)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True)

    print('Loaded model to device...')
    model = BERTNewsClf().float()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable model parameters: {total_params}')

    print('Setup Adam optimizer...')
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    print('Setup loss function...')
    loss = torch.nn.CrossEntropyLoss().to(device)

    print(f'Start training with {len(train_loader.dataset)} samples...')

    for epoch in range(1, epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        for x, y in train_loader:
            print('Starting iter')
            # extract input ids and attention mask squeeze to format (batch_size, max_bert_sequence_len)
            input_ids = x['input_ids'].squeeze(1).to(device)
            attention_mask = x['attention_mask'].squeeze(1).to(device)

            # get prediction
            y_pred, _ = model(input_ids, attention_mask)

            # compute loss
            batch_loss = loss(y_pred, y)
            epoch_loss += batch_loss
            batch_acc = (y_pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
            epoch_acc += batch_acc

            # perform gradient step
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        print(f'EPOCH: {epoch} of {epochs} with CELoss: {epoch_loss/len(train_set)} and Acc: {epoch_acc/len(train_set)}')

    print('Save model...')
    torch.save(model.state_dict(), 'transformer.t7')
