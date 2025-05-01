import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from models.efficientnet import EfficientNetModel
from data.dataset import Dataset
from utils import load_config

def load_latest_checkpoint(model, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}.")
        return None

    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by epoch number
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    model.load_state_dict(torch.load(latest_checkpoint))
    print(f"Loaded model from checkpoint: {latest_checkpoint}")
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    average_loss = total_loss / len(dataloader)
    return average_loss, accuracy, all_labels, all_predictions

if __name__ == "__main__":
    config = load_config('config.yaml')
    test_dataset = Dataset(dataset_dir=config['dataset_path'])
    #test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints_root = config['checkpoint_dir']
    subdirectories = [os.path.join(checkpoints_root, d) for d in os.listdir(checkpoints_root) if os.path.isdir(os.path.join(checkpoints_root, d))]

    evaluation_reports_dir = os.path.join(config['output_dir'], "evaluation_reports")
    os.makedirs(evaluation_reports_dir, exist_ok=True)

    for subdir in subdirectories:
        print(f"Evaluating model in directory: {subdir}")
        model = EfficientNetModel(num_classes=config['num_classes']).to(device)
        model = load_latest_checkpoint(model, subdir)

        if model is None:
            continue

        average_loss, accuracy, all_labels, all_predictions = evaluate_model(model, test_dataset.test_loader, device)
        print(f"Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.4f}")

        # Print confusion matrix and classification report
        cm = confusion_matrix(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions)

        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)

        # Save confusion matrix and classification report
        model_name = os.path.basename(subdir)
        cm_file = os.path.join(evaluation_reports_dir, f"{model_name}_confusion_matrix.txt")
        report_file = os.path.join(evaluation_reports_dir, f"{model_name}_classification_report.txt")

        with open(cm_file, "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm))

        with open(report_file, "w") as f:
            f.write("Classification Report:\n")
            f.write(report)
