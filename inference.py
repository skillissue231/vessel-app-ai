
import torch
from model.unet import get_unet_model
import numpy as np

def predict(image_tensor, model_path, encoder="resnet18", device="cpu"):
    # Load model
    model = get_unet_model(encoder, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(device)  # add batch dimension
        output = model(input_tensor)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
        return prediction
