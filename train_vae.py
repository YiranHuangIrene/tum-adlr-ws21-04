import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from VAE import *
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
def loss_function(x_hat,x,mean,log_var):
    reconstruction_loss = F.mse_loss(x_hat, x)
    kld_loss = - 0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp())
    loss = reconstruction_loss +kld_loss
    return loss.float()
if __name__ == '__main__':
    LEARNING_RATE = 1e-4
    BCE_loss = nn.BCELoss()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder(1400,512,256)
    decoder = Decoder(256,512,1400)
    model = TrajectoryVAE(1400,512,256).to(DEVICE)
    optimizer = Adam(model.parameters(),lr=LEARNING_RATE)
    full_dataset = Trajectory()
    train_size = 3000
    val_size = 1000
    test_size = 1000
    dataset_train, dataset_val ,dataset_test = torch.utils.data.random_split(full_dataset, [train_size, val_size,test_size])
    train_dataloader = DataLoader(dataset=dataset_train,batch_size=200,shuffle=True)
    val_dataloader = DataLoader(dataset=dataset_val,batch_size=200,shuffle=True)
    test_dataloader = DataLoader(dataset=dataset_test,batch_size=16,shuffle = True)
    print("Start training VAE...")
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer = SummaryWriter('runs/VAE/' + TIMESTAMP)
    EPOCHS = 1000
    for epoch in range(EPOCHS):
        overall_loss = 0
        model.train()
        for i, x in enumerate(train_dataloader):
            x = x.to(DEVICE).float()
            x_hat, mean, log_var = model(x)
            optimizer.zero_grad()
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        overall_loss/=15
        model.eval()
        overall_val_loss = 0
        for i,x, in enumerate(val_dataloader):
            with torch.no_grad():
                x = x.to(DEVICE).float()
                x_hat, mean, log_var = model(x)
                loss = loss_function(x, x_hat, mean, log_var)
                overall_val_loss += loss.item()
        overall_val_loss/=5
        print('Epoch{}/{}, training loss is:{},validation loss is:{}'.format(epoch,EPOCHS,overall_loss,overall_val_loss))
        writer.add_scalar('Training loss', overall_loss, epoch)
        writer.add_scalar('Validation loss', overall_val_loss, epoch)
