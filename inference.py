import torch
from torchvision import transforms
from PIL import Image
from models import EmoNeXtPP
from transformers import CLIPProcessor, CLIPModel


def load_model(checkpoint_path, num_classes=7):
    model = EmoNeXtPP(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def predict(model, image_path, emotion_texts):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(text=emotion_texts, return_tensors="pt", padding=True)
    text_embeds = clip_model.get_text_features(**inputs)

    with torch.no_grad():
        logits, image_embeds, _, cosine_sim = model(image, text_embeds)
        predicted_class = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    return emotion_texts[predicted_class], probs


if __name__ == '__main__':
    model = load_model("best_model.pth")
    emotion_texts = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    prediction, probs = predict(model, "test.jpg", emotion_texts)
    print(f"Predicted Emotion: {prediction}")
    print(f"Class Probabilities: {probs}")
