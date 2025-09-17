# model_utils.py
import torch
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
from config import TARGET_SIZE, class_names, gradable_classes, ungradable_reasons, MODEL_PATH

# Load model
def load_model(model_path=MODEL_PATH, num_classes=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

# Preprocess image
def preprocess_image(img, target_size=TARGET_SIZE):
    w, h = img.size
    if w != h:
        if w > h:
            delta = w - h
            padding = (0, delta//2, 0, delta - delta//2)
        else:
            delta = h - w
            padding = (delta//2, 0, delta - delta//2, 0)
        img = ImageOps.expand(img, padding, fill=(0,0,0))
    img = img.resize((target_size, target_size))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Predict class
def predict(model, device, img_tensor):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        pred_class = class_names[pred_idx.item()]
        confidence = conf.item()
        if pred_class in gradable_classes:
            gradable_status = "Gradable"
            reason = None
        else:
            gradable_status = "Ungradable"
            reason = ungradable_reasons.get(pred_class, "Quality issue")
    return pred_class, confidence, gradable_status, reason


def generate_saliency(model, image, device="cpu", cmap_name="jet", overlay=True, alpha=0.8):
    """
    Generate a saliency map for an image and return a colorized map or overlay.
    """
    model.eval()
    
    # Preprocess (assuming you already have preprocess_image)
    input_tensor = preprocess_image(image).to(device)
    input_tensor.requires_grad_()

    # Forward pass
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward pass for gradient wrt predicted class
    score = output[0, pred_class]
    score.backward()

    # Get gradients and convert to numpy
    saliency = input_tensor.grad.data.abs().max(dim=1)[0].cpu().numpy()[0]

    # Normalize to [0,1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    saliency_colored = cmap(saliency)[:, :, :3]  # RGB only
    saliency_colored = (saliency_colored * 255).astype(np.uint8)
    saliency_colored = Image.fromarray(saliency_colored).resize(image.size)

    if overlay:
        # Blend with original image
        overlay_img = Image.blend(image.convert("RGB"), saliency_colored, alpha)
        return overlay_img
    else:
        return saliency_colored