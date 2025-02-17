import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second convolutional block
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third convolutional block
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fourth convolutional block
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fifth convolutional block
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0), 
            nn.Sigmoid()  # Binary classification for real/fake
        )

    def forward(self, x):
        return self.model(x)




# import torch
# import torch.nn as nn

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
#             nn.Sigmoid()
#         )
#         resnet = getattr(self.model, 'resnet18')(pretrained=True)
#         self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
#         self.fc = nn.Linear(512, 1) 

#     def forward(self, x):
#         features = self.feature_extractor(x).view(x.size(0), -1)
#         validity = self.fc(features)
#         return validity




# class Discriminator(nn.Module):
#     def __init__(self, input_nc, resnet_type='resnet18', pretrained=True):
#         super(Discriminator, self).__init__()
#         resnet = getattr(models, resnet_type)(pretrained=pretrained)
#         self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
#         self.fc = nn.Linear(512, 1) 

#     def forward(self, x):
#         features = self.feature_extractor(x).view(x.size(0), -1)
#         validity = self.fc(features)
#         return validity

