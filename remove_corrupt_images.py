
import os
import cv2

def remove_corrupt_images(folder_path):
    removed = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)
        if image is None:
            print(f"❌ Siliniyor: {filename}")
            os.remove(file_path)
            removed += 1
    print(f"✅ {removed} bozuk görsel silindi.")

# Retina ve CAM görselleri temizleniyor
remove_corrupt_images("/Users/ahmet/Desktop/cam-vessel-ai/data/retina/images")
remove_corrupt_images("/Users/ahmet/Desktop/cam-vessel-ai/data/cam/images")
