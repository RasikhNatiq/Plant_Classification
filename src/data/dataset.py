import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

class Dataset:
    def __init__(self, dataset_dir, transform=None, batch_size=32, train_ratio=0.8):
        self.dataset_dir = dataset_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.train_loader = None
        self.test_loader = None
        self.input_shape = (3, 224, 224)
        self.num_classes = 0
        self.load_dataset()

    def load_dataset(self):
        
        # Load the dataset using ImageFolder
        full_dataset = datasets.ImageFolder(root=self.dataset_dir, transform=self.transform)
        self.num_classes = len(full_dataset.classes)
        print("Classes loaded:", full_dataset.classes)

        # Split into train and test sets
        train_size = int(self.train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Print dataset information
        print(f"Total images: {len(full_dataset)}")
        print(f"Training images: {len(train_dataset)}")
        print(f"Testing images: {len(test_dataset)}")
        print("Input shape:", self.input_shape)
        print("Number of classes:", self.num_classes)