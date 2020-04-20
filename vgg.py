import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inchannel = 1
        self.extractor = self._make_layer(config)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def _make_layer(self, config):
        layers = []
        for x in config:
            if x == 'P':
                layers.append(nn.MaxPool2d(kernel_size=2))
            else:
                layers.append(nn.Conv2d(self.inchannel, x, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                self.inchannel = x

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.extractor(x).view(-1, 1, 512)
        out = self.classifier(out)

        return out