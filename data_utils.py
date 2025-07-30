import os
from PIL import Image
from torch.utils.data import Dataset


class FERImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        self._prepare_dataset()

    def _prepare_dataset(self):
        classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_folder = os.path.join(self.root_dir, cls_name)
            if os.path.isdir(cls_folder):
                for img_file in os.listdir(cls_folder):
                    if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
                        self.image_paths.append(os.path.join(cls_folder, img_file))
                        self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
