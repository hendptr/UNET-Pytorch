import torch 
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.ConvulutionLayers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3,3)),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.ConvulutionLayers(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.ConvulutionLayers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (2,2), 2)
        )
    def forward(self, x):
        return self.ConvulutionLayers(x)


def crop_img(tensor, tensor_target):
    tensor_size = tensor.size()[2]
    target_size = tensor_target.size()[2]
    size_s = tensor_size - target_size
    size_s = size_s // 2
    return tensor[:, :, size_s:tensor_size-size_s, size_s:tensor_size-size_s]

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

        self.downsample_1 = Conv(1, 64)
        self.downsample_2 = Conv(64, 128)
        self.downsample_3 = Conv(128, 256)
        self.downsample_4 = Conv(256, 512)
        self.downsample_5 = Conv(512, 1024)

        self.max_pool_2x2 = nn.MaxPool2d((2,2), 2)

        self.uptrans_1 = UpConv(1024, 512)
        self.uptrans_2 = UpConv(512, 256)
        self.uptrans_3 = UpConv(256, 128)
        self.uptrans_4 = UpConv(128, 64)
        self.uptrans_5 = UpConv(64, 1)

        self.upsample_1 = Conv(1024, 512)
        self.upsample_2 = Conv(512, 256)
        self.upsample_3 = Conv(256, 128)
        self.upsample_4 = Conv(128, 64)
        self.out = nn.Conv2d(64, 2, 1)
    def forward(self, x):
        #Downsample
        d1 = self.downsample_1(x)
        d1_c = d1
        d1 = self.max_pool_2x2(d1)
        #print(d1.size())

        d2 = self.downsample_2(d1)
        d2_c = d2
        d2 = self.max_pool_2x2(d2)

        d3 = self.downsample_3(d2)
        d3_c = d3
        d3 = self.max_pool_2x2(d3)
        
        d4 = self.downsample_4(d3)
        d4_c = d4
        d4 = self.max_pool_2x2(d4)

        d5 = self.downsample_5(d4)

        #Upsample
        d6 = self.uptrans_1(d5)
        d6_cn = crop_img(d4_c, d6)
        d6_C = torch.cat([d6, d6_cn], 1)

        d8 = self.upsample_1(d6_C)

        d9 = self.uptrans_2(d8)
        d9_cn = crop_img(d3_c, d9)
        d9_C = torch.cat([d9, d9_cn], 1)
        
        d10 = self.upsample_2(d9_C)

        d11 = self.uptrans_3(d10)
        d11_cn = crop_img(d2_c, d11)
        d11_C = torch.cat([d11, d11_cn], 1)

        d12 = self.upsample_3(d11_C)

        d13 = self.uptrans_4(d12)
        d13_cn = crop_img(d1_c, d13)
        d13_C = torch.cat([d13, d13_cn], 1)

        d14 = self.upsample_4(d13_C)

        out = self.out(d14)
        return out

if __name__ == '__main__':
    model = UNET()
    x_input = torch.randn(1, 1, 572, 572)
    y = model(x_input)
    print(y.size())
