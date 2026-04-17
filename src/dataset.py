import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FASDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images. Expected structure:
                               data_dir/0_real/
                               data_dir/1_spoof/
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load real images (label 0)
        real_dir = os.path.join(data_dir, "0_real")
        if os.path.exists(real_dir):
            for fname in os.listdir(real_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(real_dir, fname))
                    self.labels.append(0)

        # Load spoof images (label 1)
        spoof_dir = os.path.join(data_dir, "1_spoof")
        if os.path.exists(spoof_dir):
            for fname in os.listdir(spoof_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(spoof_dir, fname))
                    self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(is_train=True):
    """Returns torchvision transforms for training or evaluation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

# Helper to generate a dummy dataset for end-to-end sandbox testing
def generate_mock_dataset(base_dir, num_samples_per_class=50):
    for split in ['train', 'val', 'test']:
        for label, class_name in [(0, '0_real'), (1, '1_spoof')]:
            dir_path = os.path.join(base_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
            for i in range(num_samples_per_class):
                # Create random noise image colored to be distinct
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                if label == 0:
                    img[:, :, 0] = 200 # Red-ish for real
                else:
                    img[:, :, 2] = 200 # Blue-ish for spoof
                cv2.imwrite(os.path.join(dir_path, f"img_{i}.jpg"), img)
