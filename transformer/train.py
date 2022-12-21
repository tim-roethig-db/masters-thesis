import torch

from model import BERTNewsClf


if __name__ == '__main__':
    batch_size = 2
    lr = 0.01
    epochs = 10

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('Loaded model to device...')
    model = BERTNewsClf()
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

    for epoch in range(1, epochs+1):
        print(f'EPOCH: {epoch} of {epochs}')
        model_in = "Replace me by any text you'd like."
        model_out = model.forward(model_in)
        print(model_out)