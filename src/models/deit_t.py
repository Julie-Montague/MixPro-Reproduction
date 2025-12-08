# src/models/deit_t.py
from timm import create_model

def build_deit_t(num_classes=1000, img_size=224, drop_path_rate=0.1, pretrained=False):
    """
    Build DeiT-Tiny (deit_tiny_patch16_224)
    """
    model = create_model(
        'deit_tiny_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    return model
