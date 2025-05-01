import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import Dataset
from models.efficientnet import EfficientNetModel
from utils import plot_training_curves
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(model, train_loader, criterion, optimizer, device, epochs, patience=5, checkpoint_dir="checkpoints"):
    model.train()
    history = {"loss": [], "accuracy": []}
    best_loss = float('inf')
    patience_counter = 0

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss /= len(train_loader)
        accuracy = correct / total
        history["loss"].append(epoch_loss)
        history["accuracy"].append(accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"custom_cnn_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    return history

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    config = load_config('config.yaml')
    batch_size = config['batch_size']
    dataset = Dataset(dataset_dir=config['dataset_path'],batch_size=batch_size)

    # Experiment with different optimizers
    optimizers = ['sgd', 'rmsprop', 'adagrad']

    for opt_name in optimizers:
        print(f"Training with optimizer: {opt_name}")
        model = EfficientNetModel(num_classes=config['num_classes']).to(device)  # Reinitialize model for each optimizer
        if opt_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        elif opt_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
        elif opt_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'])
        elif opt_name == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=config['learning_rate'])

        
        criterion = nn.CrossEntropyLoss()
    
        epochs = config['num_epochs']
        checkpoint_dir = config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"custom_cnn_model_{opt_name}.pth")

        # Train the model
        history = train_model(model, dataset.train_loader, criterion, optimizer, device, epochs, checkpoint_dir=checkpoint_path)

        # Plot training curves
        plot_training_curves(history, model_name = 'EfficientNet', optimizer_name = opt_name, batch_size = batch_size, output_dir="plots")

    
    # Experiment with different batch sizes
    opt_name = 'adam'
    batch_sizes = [32, 64, 128, 256]

    for batch_size in batch_sizes:

        dataset = Dataset(dataset_dir=config['dataset_path'],batch_size=batch_size)
        model = EfficientNetModel(num_classes=config['num_classes']).to(device)  # Reinitialize model for each batch size

        print(f"Training with Batch Size: {batch_size}")
        if opt_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        elif opt_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
        elif opt_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'])
        elif opt_name == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=config['learning_rate'])

        
        criterion = nn.CrossEntropyLoss()
    
        epochs = config['num_epochs']
        checkpoint_dir = config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"custom_cnn_model_{opt_name}_{batch_size}.pth")

        # Train the model
        history = train_model(model, dataset.train_loader, criterion, optimizer, device, epochs, checkpoint_dir=checkpoint_path)

        # Plot training curves
        plot_training_curves(history, model_name = 'EfficientNet', optimizer_name = opt_name, batch_size = batch_size, output_dir="plots")

if __name__ == "__main__":
    main()