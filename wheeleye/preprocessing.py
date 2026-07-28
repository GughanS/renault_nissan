import torchvision.transforms as T

# Standard ImageNet normalization for pretrained backbones (MobileNetV3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(image_size, augment=False):
    """
    Returns the exact transformation pipeline used for training.
    """
    transforms = [
        T.Resize(image_size),
    ]
    
    if augment:
        transforms.extend([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return T.Compose(transforms)

def get_inference_transforms(image_size):
    """
    Returns the exact transformation pipeline used for inference.
    Guaranteed to match the training distribution perfectly.
    """
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
