import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from filters import extract_noise_residual

class ForgeryDataset(Dataset):
    def __init__(self, root_dir, transform=None, preprocess_noise=True):
        # """
        # root_dir:
        #     - real/
        #     - forged/
        # """
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess_noise = preprocess_noise

        self.classes = ['real', 'forged']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg','.jpeg','.png')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = np.array(image)

        if self.preprocess_noise:
            
            residual = extract_noise_residual(image)  
            residual = residual[np.newaxis, :, :]
            return residual, label
        else:
            from torchvision.transforms import ToTensor
            return ToTensor()(image), label
