def load_config(config_path):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def plot_training_curves(history, model_name, optimizer_name, batch_size, output_dir="plots"):
    import os
    import matplotlib.pyplot as plt
    
    # Create directory for saving plots
    plot_dir = os.path.join(output_dir, f"{model_name}_{optimizer_name}_batch{batch_size}")
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.title("Accuracy")
    plt.legend()
    accuracy_plot_path = os.path.join(plot_dir, "accuracy.png")
    plt.savefig(accuracy_plot_path)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train Loss")
    plt.title("Loss")
    plt.legend()
    loss_plot_path = os.path.join(plot_dir, "loss.png")
    plt.savefig(loss_plot_path)
    
    plt.close()