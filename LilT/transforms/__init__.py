from .pixelbert import (
    pixelbert_transform,
    pixelbert_transform_randaug,
)
from torchvision import transforms
from PIL import Image

_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]


def my_transform(image_size):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    tr = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return [tr]