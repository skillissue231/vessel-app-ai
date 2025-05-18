
from PIL import Image
import os

def resize_masks(img_dir, mask_dir):
    mismatch_count = 0
    for f in os.listdir(img_dir):
        img_path = os.path.join(img_dir, f)
        mask_path = os.path.join(mask_dir, f)
        if os.path.exists(mask_path):
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            if img.size != mask.size:
                resized_mask = mask.resize(img.size, Image.NEAREST)
                resized_mask.save(mask_path)
                print(f"✔️ Resized mask: {f} to {img.size}")
                mismatch_count += 1
    print(f"✅ {mismatch_count} mask(s) resized in {mask_dir}")

# CAM dataset
resize_masks("/Users/ahmet/Desktop/cam-vessel-ai/data/cam/images",
             "/Users/ahmet/Desktop/cam-vessel-ai/data/cam/masks")

# Retina dataset
resize_masks("/Users/ahmet/Desktop/cam-vessel-ai/data/retina/images",
             "/Users/ahmet/Desktop/cam-vessel-ai/data/retina/masks")
