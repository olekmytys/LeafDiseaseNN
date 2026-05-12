import time
import torch
from torch import nn
import torch.nn.functional as F
from dataset import ImageDataset, FourierSharpening
from dataloader import train_loader, validation_loader, class_names
from sklearn.metrics import f1_score
from nmodel import LeafCNN

import torch.nn.utils.prune as prune
# ── Ustawienia ────────────────────────────────────────────────────────────────
MODEL_SAVE    = 'model.pth'
EPOCHS        = 100
LEARNING_RATE = 0.0005
NUM_CLASSES   = 2
DEVICE        = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        #inputs are logits, targets are class idxses
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pc = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pc)**self.gamma * ce_loss
        return focal_loss.mean()


model = LeafCNN().to(DEVICE)
#n_unhealthy = sum(1 for _, l in train_loader.dataset.dataset.img_labels if l == 0)
#n_healthy   = sum(1 for _, l in train_loader.dataset.dataset.img_labels if l == 1)
#weight = torch.tensor([1.0, n_unhealthy / n_healthy]).to(DEVICE)
#criterion = nn.CrossEntropyLoss(weight=weight)
#criterion = nn.CrossEntropyLoss()
criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)


# 🚂🚃🚃🚃
def train():
    start_time = time.time()
    best_val = 0.0

    for epoch in range(EPOCHS):

        # ── Trening ───────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset) * 100

        # ── Walidacja ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct = 0.0, 0
        correct_per_class = torch.zeros(NUM_CLASSES)
        total_per_class   = torch.zeros(NUM_CLASSES)
        y_true, y_pred    = [], []

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

                predictions = outputs.argmax(1)
                y_pred.extend(predictions.cpu().tolist())
                y_true.extend(labels.cpu().tolist())

                for label, prediction in zip(labels, predictions):
                    total_per_class[label] += 1
                    if label == prediction:
                        correct_per_class[label] += 1

        val_loss /= len(validation_loader.dataset)
        val_acc = val_correct / len(validation_loader.dataset) * 100
        f1 = f1_score(y_true, y_pred, average=None)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1:2d} out of {EPOCHS} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.1f}% | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.1f}% | ")

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print("  Dokładność i F1 per klasa:")
            for idx, name in enumerate(class_names): #enumerate(["Unhealthy","Healthy"]):
                if total_per_class[idx] > 0:
                    acc = correct_per_class[idx] / total_per_class[idx] * 100
                    print(f"    {name}: acc={acc:.1f}%, f1={f1[idx]:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  ✓ Model saved (val acc: {val_acc:.1f}%)")

    print(f"\n🚉 END, best_val: {best_val:.1f}%")
    print("--- %s seconds ---" % (time.time() - start_time))
    # ── Pruning ───────────────────────────────────────────────────────────
    #model.load_state_dict(torch.load(MODEL_SAVE))  # najlepszy checkpoint

    #for module in [model.conv1, model.conv2, model.conv3, model.fc1]:
    #    prune.l1_unstructured(module, name='weight', amount=0.3)

    #model.load_state_dict(torch.load(MODEL_SAVE))  # teraz klucze się zgadzają

    #total, zeroed = 0, 0
    #for module in [model.conv1, model.conv2, model.conv3, model.fc1]:
    #    total += module.weight.numel()
    #    zeroed += (module.weight == 0).sum().item()
    #print(f"Wyzerowane wagi: {zeroed}/{total} ({100*zeroed/total:.1f}%)")

    #model.eval()
    #val_correct = 0
    #with torch.no_grad():
    #    for images, labels in validation_loader:
    #        images, labels = images.to(DEVICE), labels.to(DEVICE)
    #        val_correct += (model(images).argmax(1) == labels).sum().item()
    #print(f"Accuracy po pruningu: {val_correct / len(validation_loader.dataset) * 100:.1f}%")

    #torch.save(model.state_dict(), 'model_pruned.pth')
    return best_val, f1

if __name__ == "__main__":
    train()
