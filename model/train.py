from tqdm import tqdm
from torch import optim
import torch.nn.functional as F

def train(model):
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
                output = model(x, y, output_window, 0.6).to(device)
                loss = criterion(output, y)
                #loss = criterion(output, y) + 1e-9
                #print(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                total_loss += loss.cpu().item()
            tr.set_postfix(loss="{0:.5f}".format(total_loss/len(dataloader)))