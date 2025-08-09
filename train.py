import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import ForgeryDataset
from model import ForgeryCNN
from utils import accuracy, plot_training, plot_confusion_matrix
from config import Config
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def get_transforms(cfg):
    return transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
    ])

def train():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    transform = get_transforms(cfg)
    full_dataset = ForgeryDataset(cfg.data_dir, transform=transform, preprocess_noise=True)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    model = ForgeryCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = np.inf
    epochs_no_improve = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(cfg.num_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
        model.train()
        running_loss = 0
        running_acc = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_acc += accuracy(outputs, labels) * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_acc / len(train_loader.dataset)

        model.eval()
        running_loss = 0
        running_acc = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_acc += accuracy(outputs, labels) * inputs.size(0)

                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # probability of "forged" class
                preds = outputs.argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_acc / len(val_loader.dataset)

        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        roc_auc = roc_auc_score(all_labels, all_probs)

        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), cfg.model_save_path)
            print(f"Model saved at epoch {epoch+1}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                print("Early stopping triggered.")
                break

    print("\nTraining complete.")
    plot_training(history)
    plot_confusion_matrix(all_labels, all_preds, ['real', 'forged'])

if __name__ == "__main__":
    train()
