import torch
import torch.nn as nn


def conv_layer(channel_in: int, channel_out: int, k_size: int, padding: int):
    layer = nn.Sequential(
        nn.Conv2d(channel_in, channel_out, k_size, padding=padding),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_size, pooling_stride):
    layers = [
        conv_layer(in_list[i], out_list[i], k_list[i], p_list[i])
        for i in range(len(in_list))]
    layers += [nn.MaxPool2d(pooling_size, stride=pooling_stride)]

    return nn.Sequential(*layers)


def vgg_fc_layer(in_features, out_features):
    layer = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU()
    )

    return layer


class ANNVGG16(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        
        # convolutional layers
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        # self.layer6 = vgg_fc_layer(7*7*512, 4096)
        # self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(512, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        # out = self.layer6(out)
        # out = self.layer7(out)
        out = self.layer8(out)

        return out
