import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import os

### hyperparamters

batch_size = 32

###

class StandfordCars(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        super().__init__()

        self.image_folder = image_folder
        self.transform = transform

        self.image_files = sorted(os.listdir(image_folder))
    
    def len(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x * 2 - 1) # [-1, 1]
])
dataset = StandfordCars('./stanford_cars', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def show_tensor_image(image):
    reverse_transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: (x + 1) / 2), # [0, 1]
        torchvision.transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        torchvision.transforms.Lambda(lambda t: t * 255.0),
        torchvision.transforms.ToPILImage()
    ])

    if len(image.shape) == 4:
        image = image[0]
    
    plt.savefig(reverse_transform(image))