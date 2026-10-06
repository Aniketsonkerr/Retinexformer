import os
import torch
import cv2
import numpy as np
from basicsr.models.archs.retinexformer_arch import RetinexFormer

# 1. Load Model Architecture & Weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RetinexFormer().to(device)

weights_path = 'pretrained_weights/LOL_v1.pth'
checkpoint = torch.load(weights_path, map_location=device)

if 'params' in checkpoint:
    model.load_state_dict(checkpoint['params'])
elif 'params_ema' in checkpoint:
    model.load_state_dict(checkpoint['params_ema'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# 2. Setup Directories
input_dir = 'data/CustomFaces/input'
output_dir = 'results/CustomFaces_Enhanced'
os.makedirs(output_dir, exist_ok=True)

# 3. Inference Loop
for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path).astype(np.float32) / 255.0
    
    # BGR to RGB and HWC to NCHW Tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(img_tensor)

    # Tensor to BGR Image
    output_img = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, img_name), output_img)
    print(f"Processed: {img_name}")

print(f"\nDone! Enhanced face images saved to: {output_dir}")