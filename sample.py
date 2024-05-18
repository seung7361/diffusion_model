import torch
from model import Unet
from model import LinearNoiseScheduler
from tqdm import tqdm

import torchvision
from torchvision.utils import make_grid

import os

device = "cuda:2"

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
model.eval()
model.load_state_dict(torch.load('cifar10.pth', map_location=device))

@torch.no_grad()
def sample(model, scheduler):
    x_t = torch.randn(100, 3, 28, 28).to(device)
    
    for i in tqdm(reversed(list(range(5000))), total=5000):
        noise_pred = model(x_t, torch.as_tensor(i).unsqueeze(0).to(device))

        x_t, x_0_pred = scheduler.sample_prev_timestep(x_t, noise_pred, torch.as_tensor(i).to(device))

        images = torch.clamp(x_t, -1., 1.).detach()
        images = (images + 1) / 2
        grid = make_grid(images, nrow=10)
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join('cifar', 'samples')):
            os.mkdir(os.path.join('cifar', 'samples'))
        img.save(os.path.join('cifar', 'samples', 'x0_{}.png'.format(i)))
        img.close()

scheduler = LinearNoiseScheduler(5000, 1e-4, 2e-2)
sample(model, scheduler)