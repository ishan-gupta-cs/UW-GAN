import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder with deeper layers and increased feature maps
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Downsample to 1/2
            nn.ReLU(inplace=True),
            ResBlock(64),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Downsample to 1/4
            nn.ReLU(inplace=True),
            ResBlock(128),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Downsample to 1/8
            nn.ReLU(inplace=True),
            ResBlock(256),
        )

        # Decoder with additional upsampling layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample to 1/4
            nn.ReLU(inplace=True),
            ResBlock(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample to 1/2
            nn.ReLU(inplace=True),
            ResBlock(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Upsample to original size
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Final output layer
            nn.Tanh()  # Normalize output to [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)





# import torch
# import torch.nn as nn

# class ResBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(ResBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels)
#         )

#     def forward(self, x):
#         return x + self.block(x)

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             ResBlock(64),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 3, kernel_size=3, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         return self.decoder(x)


# import torch.nn as nn
# import torchvision.models as models

# class Generator(nn.Module):
#     def __init__(self, input_nc, output_nc, resnet_type='resnet18', pretrained=True):
#         super(Generator, self).__init__()
#         resnet = getattr(models, resnet_type)(pretrained=pretrained)
#         self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
#         self.upconv = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, output_nc, 4, stride=2, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         output = self.upconv(features)
#         return output
