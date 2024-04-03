from tqdm import tqdm
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(device, model, dataloader, learning_rate, epoch, output_window):
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.0001)
    criterion = nn.L1Loss()
    torch.autograd.set_detect_anomaly(True)

    with tqdm(range(epoch)) as tr:
        for i in tr:
            total_loss = 0.0
            for x,y in dataloader:
                optimizer.zero_grad()
                x = x.to(device).float()
                y = y.to(device).float()
                output = model(x, y, output_window, 0).to(device)
                print("model output shape : ", output.shape)
                print("label shape : ", y.shape)
                loss = criterion(output, y)
                print("loss : ", loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                total_loss += loss.cpu().item()
            tr.set_postfix(loss="{0:.5f}".format(total_loss/len(dataloader)))

    return model