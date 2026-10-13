import os
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import urllib.request

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

# 1. Initialize Face Cascade with Auto-Download Fallback
cascade_filename = 'haarcascade_frontalface_default.xml'
local_cascade_path = os.path.join(BASE_DIR, cascade_filename)

if not os.path.exists(local_cascade_path):
    print(f"[Info] Downloading missing face detection model ({cascade_filename})...")
    url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{cascade_filename}"
    try:
        urllib.request.urlretrieve(url, local_cascade_path)
        print("[Info] Download complete.")
    except Exception as e:
        print(f"[Warning] Failed to download face cascade: {e}")

face_cascade = cv2.CascadeClassifier()
if os.path.exists(local_cascade_path):
    face_cascade.load(local_cascade_path)

has_detector = not face_cascade.empty()
if has_detector:
    print("[Info] Face detector loaded successfully.")
else:
    print("[Warning] Face detector failed to load. Running full-image face enhancement mode.")

# 2. Load Retinexformer Model & Weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RetinexFormer(
    in_channels=3,
    out_channels=3,
    n_feat=40,
    stage=1,
    num_blocks=[1, 2, 2]
).to(device)

weights_path = os.path.join(BASE_DIR, 'pretrained_weights', 'LOL_v1.pth')
checkpoint = torch.load(weights_path, map_location=device)

if 'params' in checkpoint:
    model.load_state_dict(checkpoint['params'])
elif 'params_ema' in checkpoint:
    model.load_state_dict(checkpoint['params_ema'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# 3. Setup Directories
input_dir = os.path.join(BASE_DIR, 'data', 'CustomFaces', 'Test', 'input')
if not os.path.exists(input_dir):
    input_dir = os.path.join(BASE_DIR, 'data', 'CustomFaces', 'input')

output_dir = os.path.join(BASE_DIR, 'results', 'CustomFaces_Enhanced')
os.makedirs(output_dir, exist_ok=True)

# 4. Inference Loop with Dynamic Tensor Padding
for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_dir, img_name)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue

    targets = []
    
    if has_detector:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        if len(faces) > 0:
            print(f"[{img_name}] Detected {len(faces)} face(s). Extracting regions...")
            for idx, (x, y, w, h) in enumerate(faces):
                pad_x, pad_y = int(w * 0.2), int(h * 0.2)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(img_bgr.shape[1], x + w + pad_x), min(img_bgr.shape[0], y + h + pad_y)
                targets.append((f"face_{idx+1}_{img_name}", img_bgr[y1:y2, x1:x2]))

    if not targets:
        targets.append((img_name, img_bgr))

    for out_filename, crop in targets:
        crop_float = crop.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(crop_float).permute(2, 0, 1).unsqueeze(0).to(device)

        # Pad spatial dimensions to a multiple of 16 to avoid UNet dimension mismatch
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        multiple = 16
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            output_tensor = model(img_tensor)

        # Crop output back to original face crop dimensions
        output_tensor = output_tensor[:, :, :h, :w]

        output_img = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir, out_filename), output_img)
        print(f"  └─ Saved enhanced output: {out_filename}")

print(f"\nExecution finished! Results saved in: {output_dir}")