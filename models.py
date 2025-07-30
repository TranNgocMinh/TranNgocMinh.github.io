import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import CLIPModel, CLIPProcessor


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FESDecoder(nn.Module):
    def __init__(self, in_features, img_size):
        super(FESDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, img_size * img_size),
            nn.Tanh()
        )
        self.img_size = img_size

    def forward(self, x):
        x = self.decoder(x)
        return x.view(-1, 1, self.img_size, self.img_size)


class EmoNeXtPP(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super(EmoNeXtPP, self).__init__()
        base_model = convnext_tiny(pretrained=True)
        self.backbone = create_feature_extractor(base_model, return_nodes={'avgpool': 'features'})
        self.stn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 53 * 53, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.se = SELayer(768)
        self.classifier = nn.Linear(768, num_classes)
        self.fes_decoder = FESDecoder(768, 48)

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.mapping = nn.Linear(768, embedding_dim)

    def stn_transform(self, x):
        xs = self.stn(x)
        xs = xs.view(-1, 10 * 53 * 53)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self, x, text_embeddings):
        x_transformed = self.stn_transform(x)
        features = self.backbone(x_transformed)['features']
        features = self.se(features)
        features = features.mean([-2, -1])  # Global average pooling
        image_embedding = self.mapping(features)

        # FES branch
        synthesized_img = self.fes_decoder(features)

        # Cosine similarity with text embeddings
        cosine_sim = F.cosine_similarity(image_embedding.unsqueeze(1), text_embeddings.unsqueeze(0), dim=-1)

        logits = self.classifier(features)
        return logits, image_embedding, synthesized_img, cosine_sim
