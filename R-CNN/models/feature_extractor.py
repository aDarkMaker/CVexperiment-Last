import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from torchvision import models  # pyright: ignore[reportMissingImports]

try:
    from torchvision.models import AlexNet_Weights  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    AlexNet_Weights = None  # type: ignore[assignment]

class AlexNetExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            try:
                alex = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)  # type: ignore[union-attr]
            except AttributeError:
                alex = models.alexnet(pretrained=True)
        else:
            alex = models.alexnet(weights=None)
        self.features = alex.features
        self.avgpool = alex.avgpool
        self.fc6 = alex.classifier[0:3]   # fc6 + relu + dropout
        self.fc7 = alex.classifier[3:6]   # fc7 + relu + dropout

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc6(x)
        x = self.fc7(x)
        return x