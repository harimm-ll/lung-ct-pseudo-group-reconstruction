from __future__ import annotations

from PIL import Image
from torchvision import transforms


def build_image_transform(image_size: int): 
    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485， 0.456， 0.406),
                std=(0.229， 0.224， 0.225),
            ),
        ]
    )


def load_rgb(path, transform):  
    with Image.open(path) as image:
        return transform(image.convert("RGB"))

