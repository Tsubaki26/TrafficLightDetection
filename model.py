from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNet101_Weights
from torchvision.models.detection import FasterRCNN


def fasterrcnn_backbone_resnet101(
    num_classes, pretrained=True, freeze_layer_on_fpn=False
):
    """
    特徴抽出
    """
    if freeze_layer_on_fpn:
        TRAINABLE_LAYERS = 0
    else:
        TRAINABLE_LAYERS = 5

    if pretrained:
        backbone = resnet_fpn_backbone(
            backbone_name="resnet101",
            weights=ResNet101_Weights.DEFAULT,
            trainable_layers=TRAINABLE_LAYERS,
        )
    else:
        backbone = resnet_fpn_backbone(
            backbone_name="resnet101",
            trainable_layers=TRAINABLE_LAYERS,
        )

    model = FasterRCNN(backbone, num_classes)

    return model
