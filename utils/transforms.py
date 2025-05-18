
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(p=0.3, shift_limit=0.05, scale_limit=0.1, rotate_limit=15),
        ToTensorV2()
    ])
