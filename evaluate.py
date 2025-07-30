import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from models import EmoNeXtPP
from data_utils import FERImageDataset
from torchvision import transforms


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits, _, _, _ = model(images, text_embeddings)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, digits=4)
    return acc, f1, report


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_dataset = FERImageDataset("./data/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = EmoNeXtPP(num_classes=7)
    model.load_state_dict(torch.load("best_model.pth"))
    model.cuda()

    from transformers import CLIPTokenizer, CLIPModel
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    texts = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to('cuda')
    text_embeddings = clip_model.get_text_features(**inputs)

    acc, f1, report = evaluate_model(model.cuda(), test_loader, 'cuda')
    print(f"Accuracy: {acc:.4f}\nF1 Score: {f1:.4f}\n")
    print(report)
