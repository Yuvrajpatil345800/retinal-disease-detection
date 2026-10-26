
from pathlib import Path

import torch
import timm
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_vit_retinal_improved.pth"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

DEVICE = torch.device("cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model file not found: {MODEL_PATH}"
        )

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=len(CLASS_NAMES),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    # Handle checkpoints saved by some training wrappers.
    if isinstance(checkpoint, dict) and any(
        key.startswith("module.") for key in checkpoint
    ):
        checkpoint = {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
        }

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image):
    if not isinstance(image, Image.Image):
        raise TypeError("Please provide a PIL image.")

    image = image.convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "probabilities": {
            name: float(probabilities[i].item())
            for i, name in enumerate(CLASS_NAMES)
        },
    }
