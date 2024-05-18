import torch
import torchvision
from tqdm import tqdm

from model import LinearNoiseScheduler
from model import Unet

import matplotlib.pyplot as plt

device = "cuda:1"
T = 5000

scheduler = LinearNoiseScheduler(T, 1e-6, 1e-2)
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

model = Unet({
    'im_channels': 3,
    'im_size': 28,
    'down_channels': [32, 64, 128, 256],
    'mid_channels': [256, 256, 128],
    'time_emb_dim': 128,
    'down_sample': [True, True, False],
    'num_down_layers': 2,
    'num_mid_layers': 2,
    'num_up_layers': 2
}).to(device)
model.train()

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

losses = []
for epoch in range(6000):
    pbar = tqdm(dataloader)
    pbar.set_description(f"Epoch: {epoch}")

    for im, _ in pbar:
        optim.zero_grad()
        im = im.float().to(device)

        noise = torch.randn_like(im).to(device)
        t = torch.randint(0, T, (im.shape[0],)).to(device)

        noisy_image = scheduler.add_noise(im, noise, t)
        noise_pred = model(noisy_image, t)

        loss = loss_fn(noise_pred, noise)
        loss.backward()
        optim.step()

        losses.append(loss.item())
        pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.4f}")


    print(f"Epoch: {epoch}, Loss: {sum(losses) / len(losses)}")

    torch.save(model.state_dict(), f"cifar10_2.pth")

plt.plot(losses)
plt.savefig('loss.png')
plt.close()