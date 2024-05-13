import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34
from collections import OrderedDict
import torch

## 네트워크 구축
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x

## 네트워크 구축
def BNReLU(num_features):
    return nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU())

# Context Semantic Branch
class Semantic_Branch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet18(pretrained); resnet.layer3 = nn.Identity()
        resnet.layer4 = nn.Identity(); resnet.avgpool = nn.Identity()
        resnet.fc = nn.Identity();
        self.model = nn.Sequential(OrderedDict([("prefix", nn.Sequential(resnet.conv1,
                                                                        resnet.bn1,
                                                                        resnet.relu,
                                                                        resnet.maxpool)),
                                                ("layer1", resnet.layer1),
                                                ("layer2", resnet.layer2),
                                                ]))
    def forward(self, x):
        x = self.model.prefix(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        return x

# Spatial Detail Branch
class Detail_Branch(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=True); resnet.layer2 = nn.Identity()
        resnet.layer3 = nn.Identity(); resnet.layer4 = nn.Identity()
        resnet.avgpool = nn.Identity(); resnet.fc = nn.Identity()
        self.model = nn.Sequential(OrderedDict([("prefix", nn.Sequential(resnet.conv1,
                                                                        resnet.bn1,
                                                                        resnet.relu,
                                                                        resnet.maxpool)),
                                                ("layer1", resnet.layer1),
                                                ]))

        self.channel_adjust = nn.Sequential(nn.Conv2d(64, 128, 1), BNReLU(128))

    def forward(self, x):
        x = self.model(x)
        x = self.channel_adjust(x)
        return x

# selective fusion module
class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_conv = nn.Sequential(nn.Conv2d(256, 128, 1),
                                    BNReLU(128),
                                    nn.Conv2d(128, 1, 1))

    def forward(self, low, high):
        high = F.interpolate(high, size=(low.size(2), low.size(3)),
                                    mode="bilinear", align_corners=False)
        attmap = torch.cat([high, low], dim=1)
        attmap = self.point_conv(attmap)
        attmap = torch.sigmoid(attmap)
        return attmap * low + high

# core structure of LFD-RoadSeg
class LFD_RoadSeg(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

        self.semantic = Semantic_Branch(pretrained=True)

        self.context1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 5), dilation=2, padding=(0, 4), groups=128),
                                        nn.Conv2d(128, 128, kernel_size=(5, 1), dilation=1, padding=(2, 0),  groups=128),
                                        nn.Conv2d(128, 128, kernel_size=1),
                                        BNReLU(128))

        self.context2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 5), dilation=1, padding=(0, 2), groups=128),
                                        nn.Conv2d(128, 128, kernel_size=(5, 1), dilation=1, padding=(2, 0),  groups=128),
                                        nn.Conv2d(128, 128, kernel_size=1),
                                        BNReLU(128))

        self.detail = Detail_Branch()
        self.fusion = Fusion()

        self.cls_head = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1),
                                        BNReLU(128),
                                        nn.Conv2d(128, 1, kernel_size=1))

    def forward(self, inputs):
        """
            Args:
                data: A dictionary containing "img", "label", and "filename"
                data["img"]: (b, c, h, w)
        """
        x_ = inputs
        x__ = F.interpolate(x_, size=(x_.size(2)//self.scale_factor, x_.size(3)//(2*self.scale_factor)),
                        mode="bilinear", align_corners=False)

        # context semantic branch
        x_1 = self.semantic(x__)
        del x__

        # aggregation module
        x_1 = self.context1(x_1) + x_1
        x_1 = self.context2(x_1) + x_1

        # spatial detail branch
        x = self.detail(x_)

        # selective fusion module
        x_1 = self.fusion(x,x_1)

        score_map = self.cls_head(x_1)
        score_map = F.interpolate(score_map, size=(x_.size(2), x_.size(3)),
                        mode="bilinear", align_corners=False)

        return score_map
    


## 네트워크 구축
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x