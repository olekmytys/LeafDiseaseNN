import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import ImageDataset, FourierSharpening
from dataloader import train_loader, validation_loader
from nmodel import *

MODEL_SAVE = 'model.pth'
EPOCHS = 20
BATCH_SIZE = 32
LAERNING_RATE = 0.001
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = LeafCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LAERNING_RATE)

# 🚂🚃🚃🚃
def train():

    best_val = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad() 
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0) # srednia strata z batchu * ilosc zdjec w nim
            train_correct += (outputs.argmax(1) == labels).sum().item() # argmax(1) -> najbardziej prawdopodobna klasa (najwieksza liczba na zdjeice)
                                                                        # outputs - tensor o kształcie batch x ilosc klas



        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset) * 100


        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(validation_loader.dataset)
        val_acc = val_correct / len(validation_loader.dataset) * 100

        print(f"Epoch {epoch + 1} out of {EPOCHS} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.1f}% | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.1f}%")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"Model saved (val acc: {val_acc:.1f}%)")

    print(f"\n🚉 END, best_val: {best_val:.1f}%")

if __name__ == "__main__":
    train()



