import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import argparse

class CreateNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_prob=0.4):
        super(CreateNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.BatchNorm1d(hidden_size//4),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.output = nn.Linear(hidden_size//4, 1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)

def load_data(data_path, test_size=0.2):
    X = pd.read_csv(os.path.join(data_path, 'X_train.csv')).values
    y = pd.read_csv(os.path.join(data_path, 'y_train.csv')).values

    # Calculate class weights
    y_flat = y.squeeze().astype(int)
    class_counts = np.bincount(y_flat)
    pos_weight = torch.tensor([class_counts[0]/class_counts[1]], dtype=torch.float32)

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    
    train_size = int((1 - test_size) * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset)-train_size])
    
    return train_dataset, val_dataset, scaler, pos_weight

def plot_metrics(history):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.show()

def train_model(data_path='./data', model_path='model.pth', 
                epochs=150, lr=0.0005, batch_size=32, hidden_size=128,
                patience=8, dropout_prob=0.4):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data with class weights
    train_dataset, val_dataset, scaler, pos_weight = load_data(data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = train_dataset[0][0].shape[0]
    model = CreateNN(input_size, hidden_size, dropout_prob).to(device)
    
    # Class-weighted loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': []
    }
    
    best_f1 = 0
    early_stop_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        all_preds, all_targets = [], []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(1), y_batch.squeeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs.squeeze(1), y_val.squeeze(1))
                epoch_val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(y_val.cpu().numpy())
        
        # Calculate metrics
        train_acc, train_f1 = calculate_metrics(all_targets, all_preds)
        val_acc, val_f1 = calculate_metrics(val_targets, val_preds)
        
        # Update history
        avg_train_loss = epoch_train_loss/len(train_loader)
        avg_val_loss = epoch_val_loss/len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'model': model.state_dict(),
                'scaler': scaler,
                'history': history
            }, model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if early_stop_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Plot training curves
    plot_metrics(history)
    print(f"\nBest Validation F1: {best_f1:.4f}")

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true).squeeze().astype(int)
    y_pred = (np.array(y_pred).squeeze() > 0.5).astype(int)
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train EnhancedNN with Class Balancing')
    parser.add_argument('--data-path', default='./data', help='Path to training data')
    parser.add_argument('--model-path', default='model.pth', help='Path to save model')
    parser.add_argument('--epochs', type=int, default=150, help='Maximum training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout probability')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        model_path=args.model_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        patience=args.patience,
        dropout_prob=args.dropout
    )