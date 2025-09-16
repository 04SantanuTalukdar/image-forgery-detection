import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from filters import extract_noise_residual

class ForgeryDataset(Dataset):
    def __init__(self, root_dir, transform=None, preprocess_noise=True):
        """
        root_dir: path to a split folder (train/validation/test)
                  which contains:
                      real/
                      fake/
        """
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess_noise = preprocess_noise

        # Keep the folder names consistent
        self.classes = ['real', 'fake']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Collect all image paths and labels
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                raise ValueError(f"Expected folder '{cls_dir}' not found.")
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(cls_dir, fname)
                    label = self.class_to_idx[cls]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        # Apply image transform if provided
        if self.transform:
            image = self.transform(image)
            if isinstance(image, Image.Image):  # still PIL, convert to NumPy
                image = np.array(image)
        else:
            image = np.array(image)

        if self.preprocess_noise:
            residual = extract_noise_residual(image)
            # Add channel dimension for single-channel residual
            residual = residual[np.newaxis, :, :]
            return residual, label
        else:
            from torchvision.transforms import ToTensor
            return ToTensor()(image), label