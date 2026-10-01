import torch
import torch.nn as nn
import config

class SharedEncoderMultiTaskNet(nn.Module):
    """
    Bifurcated 2D multi-task architecture with a shared encoder backbone,
    a segmentation decoder branch, and a 5-class cardiomyopathy classifier branch.
    """
    def __init__(self, in_c=config.IN_CHANNELS, num_seg=config.NUM_SEG_CLASSES, num_class=config.NUM_DISEASE_CLASSES):
        super(SharedEncoderMultiTaskNet, self).__init__()

        # Shared 2D Convolutional Encoder
        self.encoder = nn.Sequential(
            self._conv_block(in_c, 32),
            nn.MaxPool2d(2),
            self._conv_block(32, 64),
            nn.MaxPool2d(2),
            self._conv_block(64, 128),
            nn.MaxPool2d(2),
            self._conv_block(128, 256)
        )

        # Branch A: Segmentation Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            self._conv_block(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            self._conv_block(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            self._conv_block(32, 32),
            nn.Conv2d(32, num_seg, kernel_size=1)
        )

        # Branch B: Disease Classifier Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_class)
        )

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        latent_features = self.encoder(x)
        segmentation_mask = self.decoder(latent_features)
        
        pooled = self.global_pool(latent_features).view(x.size(0), -1)
        clinical_class = self.classifier(pooled)

        return segmentation_mask, clinical_class
