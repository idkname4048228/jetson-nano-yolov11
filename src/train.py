import os

from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from model import SimpleCNN
from dataloader import ImageLoader

random_seed = 42
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, num_epochs, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1:2d}/{num_epochs:02d}]: train", leave=False)
        for images, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loader_tqdm.set_postfix(train_loss=loss.item())

        val(model, val_loader, epoch, num_epochs, criterion, running_loss)

def val(model, val_loader, epoch, num_epochs, criterion, running_loss):
    model.eval()
    correct, total = 0, 0
    val_loss = 0

    val_loader_tqdm = tqdm(
        val_loader, 
        desc=f"Epoch [{epoch+1:2d}/{num_epochs:02d}]: val", 
        leave=True, 
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}")
    with torch.no_grad():
        for index, (images, labels) in enumerate(val_loader_tqdm):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
            if index == len(val_loader_tqdm) - 1:
                avg_train_loss = running_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100 * correct / total

                val_loader_tqdm.set_postfix_str(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:6.2f}%")


def test(model, test_loader):
    model.eval()  # 設定為評估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = f'{100 * correct / total:.2f}' 
    save_model(model, test_acc)
    
    print(f"Test Accuracy: {test_acc}%")

def save_model(model, acc, folder='../output5/'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 取得當前時間
    timestamp = datetime.now().strftime("%m-%d")
    
    # 組合檔名
    filename = f"{acc}_{timestamp}.pth"
    filepath = os.path.join(folder, filename)
    
    # 保存模型
    torch.save(model, filepath)
    print(f'model save as {filepath}')


if __name__ == '__main__':
    dataset = ImageLoader('../../img5/', batch_size=4)
    train_loader, val_loader, test_loader = dataset.get_loaders()

    num_epochs = 10
    model = SimpleCNN().to(device)
    train(model, num_epochs, train_loader, val_loader)
    test(model, test_loader)


