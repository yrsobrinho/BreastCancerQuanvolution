import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()]) 
dataset = datasets.ImageFolder("/home/eflammere/BreastCancerQuanvolution/Datasets/BCDR/png", transform=transform)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

mean = torch.zeros(3) 
std = torch.zeros(3)

total_samples = 0

for images, _ in loader:
    batch_samples = images.size(0)  
    images = images.view(batch_samples, 3, -1) 
    mean += images.mean(dim=[0, 2]) * batch_samples
    std += images.std(dim=[0, 2]) * batch_samples
    total_samples += batch_samples

mean /= total_samples
std /= total_samples

print(f"Mean: {mean}")
print(f"Std: {std}")