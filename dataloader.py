import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from dataset import ImageDataset, FourierSharpening

transform = v2.Compose([
    v2.Resize((128, 128)), #resize
    v2.RandomHorizontalFlip(0.5),
    v2.RandomRotation(30),
    v2.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
            ),
    v2.ToDtype(torch.float32, scale=True),
    FourierSharpening(),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#dir needs change
dataset = ImageDataset(dir="./prracownia/Potato Leaf Disease Dataset in Uncontrolled Environment/", transform=transform)

validation_size = int(len(dataset) * 0.2)
train_size = len(dataset) - validation_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)