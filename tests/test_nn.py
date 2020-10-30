import torch
import auraloss

from data import LibriMixDataset

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Simple 3 layer CNN
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 16, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 1, kernel_size=3, padding=1, bias=False),
            torch.nn.Tanh())

    def forward(self, x):
        return self.model(x)

net = CNN()
criterion = auraloss.time.LogCoshLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# setup dataset
dataset = LibriMixDataset("data/MiniLibriMix")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for bidx, batch in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        s1, _, noise, _ = batch
        s1_noisy = s1 + noise

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        s1_clean = net(s1_noisy)
        loss = criterion(s1_clean, s1)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        print('[%d, %5d] loss: %.3e' %
                (epoch + 1, bidx + 1, running_loss / 2000))
        running_loss = 0.0

print('Finished Training')