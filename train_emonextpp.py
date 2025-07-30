import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models import EmoNeXtPP
from losses import TotalLoss
from transformers import CLIPTokenizer, CLIPModel


class FERDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Fake data for example
train_loader = DataLoader(FERDataset(['dummy.jpg']*32, [0]*32, transform), batch_size=8)

model = EmoNeXtPP(num_classes=7).cuda()
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()

emotion_texts = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
inputs = tokenizer(emotion_texts, return_tensors="pt", padding=True).to('cuda')
text_embeds = clip_model.get_text_features(**inputs)

neutral_idx = emotion_texts.index("neutral")
neutral_embeds = text_embeds[neutral_idx].unsqueeze(0).repeat(8, 1)

weights = torch.ones(7).cuda()
criterion = TotalLoss(weights=weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

model.train()
for epoch in range(10):
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()

        logits, image_embeds, recon_img, cosine_sim = model(images, text_embeds)
        attention_map = torch.ones_like(logits)  # Placeholder
        target_img = images[:, 0:1, :, :]  # Grayscale approximation

        loss = criterion(logits, labels, image_embeds, text_embeds[labels], neutral_embeds,
                         recon_img, target_img, attention_map)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
