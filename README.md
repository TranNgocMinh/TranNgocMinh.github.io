# EmoNeXt++-Tiny for FER2013

This repository implements **EmoNeXt++-Tiny**, a lightweight but powerful facial expression recognition model that extends ConvNeXt with STN, SE block, CLIP-based embedding supervision, and facial expression synthesis (FES).

## 📦 File Structure

```bash
├── models.py               # EmoNeXt++-Tiny architecture
├── losses.py               # Custom loss functions
├── train_emonextpp.py      # Training loop
├── inference.py            # Inference from image
├── evaluate.py             # Accuracy, F1 evaluation
├── data_utils.py           # Dataset class for FER2013-style folder
└── README.md               # Usage guide
```

## 🗂️ Dataset Format (FER2013)
Prepare the FER2013 dataset in the following format:

```
./data/train/
├── angry/
├── disgust/
├── fear/
├── happy/
├── sad/
├── surprise/
└── neutral/

./data/test/  (same structure)
```

Each subfolder contains images belonging to that emotion class.

## 🏋️‍♀️ Training
```bash
python train_emonextpp.py
```
This will train the model with CLIP supervision and self-contrastive loss.

## 🧪 Evaluation
```bash
python evaluate.py
```
Will output accuracy, F1 score, and classification report on `./data/test`.

## 🔍 Inference
```bash
python inference.py
```
Ensure you have a test image (`test.jpg`) and a trained model at `best_model.pth`.

## 💡 Features
- ConvNeXt-Tiny backbone
- Spatial Transformer Network (STN)
- Squeeze-and-Excitation block
- CLIP-based embedding for better generalization
- Facial Expression Synthesis for class imbalance
- Multi-loss strategy: CE + Contrastive + Recon + Attention regularization

## 🧠 Citation & Base
Inspired by:
- EmoNeXt [https://github.com/yelboudouri/EmoNeXt]
- CLIP [https://openai.com/clip]

## 🛠 Requirements
- Python 3.10
- PyTorch >= 2.1.1
- torchvision
- transformers
- scikit-learn

## 📝 License
MIT
