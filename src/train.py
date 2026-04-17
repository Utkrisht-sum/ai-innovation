import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import FASDataset, get_transforms
from model import FASModel, get_device

def train(epochs=3, batch_size=16, lr=1e-4, data_dir='data'):
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    train_dataset = FASDataset(os.path.join(data_dir, 'train'), transform=get_transforms(is_train=True))
    val_dataset = FASDataset(os.path.join(data_dir, 'val'), transform=get_transforms(is_train=False))

    if len(train_dataset) == 0:
        print("No training data found. Please run the dummy data generator or place data in data/train.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = FASModel(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0

    print("Starting training...")
    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation Phase
        model.eval()
        running_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = running_loss / total if total > 0 else 0
        epoch_val_acc = correct / total if total > 0 else 0
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pt')
            print("Saved new best model!")

    # Plot metrics
    os.makedirs('plots', exist_ok=True)

    # Loss Curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('plots/loss_curve.png')
    plt.close()

    # Accuracy Curve
    plt.figure()
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig('plots/accuracy_curve.png')
    plt.close()

    print("Training complete. Plots saved to plots/")

if __name__ == '__main__':
    # Default execution for sandbox verification
    train(epochs=3)
