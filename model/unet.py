import segmentation_models_pytorch as smp

def get_unet_model(encoder_name="resnet18", num_classes=1, pretrained=True, attention_type="scse"):
    """
    Returns a U-Net with optional attention gates.

    Args:
        encoder_name (str): name of the encoder, e.g., "resnet18", "resnet34", etc.
        num_classes (int): number of output classes/channels.
        pretrained (bool): whether to use ImageNet pretrained weights.
        attention_type (str or None): type of attention gate: "scse", "cbam", or None.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
        attention_type=attention_type
    )
    return model
