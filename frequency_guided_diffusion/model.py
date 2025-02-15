import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_wavelets as pw
from torch.nn import init
import torch
import torch.nn as nn

class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel, stride=stride)

    def forward(self, x):
        return self.upsample(x)
    
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, stride=stride,padding=padding, bias=False),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return 0.1*self.conv_block(x) + self.conv_skip(x)
    
class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[3, 6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims+out_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims+out_dims*2, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block4 = nn.Sequential(
            nn.Conv2d(
                in_dims+out_dims*3, out_dims, 3, stride=1, padding=rate[3], dilation=rate[3]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.output = nn.Conv2d(in_dims+out_dims*4, out_dims, 1)

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(torch.cat((x, x1), dim=1))
        x3 = self.aspp_block3(torch.cat((x, x1, x2), dim=1))
        x4 = self.aspp_block4(torch.cat((x, x1, x2, x3), dim=1))
        out = torch.cat([x, x1, x2, x3, x4], dim=1)
        return self.output(out)
def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(
        out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        :,
        max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
        max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0),
    ]

    # out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    # out = out.permute(0, 2, 3, 1)

    return out[:, :, ::down_y, ::down_x]

def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h

    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

class HaarTransform(nn.Module):
    def __init__(self, in_channels=1,four_channels = True,levels =1):
        super().__init__()


        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)
        self.four_channels = four_channels

        if levels > 1 and not four_channels :
            self.next_level = HaarTransform(in_channels,four_channels,levels-1)
        else :
            self.next_level = None

    def forward(self, input):

        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)

        if self.next_level != None :
            ll = self.next_level(ll)

        if self.four_channels : # 运行这儿
            return torch.cat((ll, lh, hl, hh), 1)
        else :
            return torch.cat((torch.cat((ll,lh),-2),torch.cat((hl,hh),-2)),-1)
        
class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels,four_channels = True,levels = 1):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)

        self.four_channels = four_channels

        if levels > 1 and not four_channels :
            self.next_level = InverseHaarTransform(in_channels,four_channels,levels-1)
        else :
            self.next_level = None

    def forward(self, input):
        if self.four_channels :
            ll, lh, hl, hh = input.chunk(4, 1)
        else :
            toprow,bottomrow = input.chunk(2,-1)
            ll,lh = toprow.chunk(2,-2)
            hl,hh = bottomrow.chunk(2,-2)

        if self.next_level != None :
            ll = self.next_level(ll)

        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))

        return ll + lh + hl + hh     
        
class ResUnet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512], J=1):
        super(ResUnet, self).__init__()

        self.dwt = HaarTransform(1)
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.squeeze_excite0 = Squeeze_Excite_Block(filters[0])

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], kernel_size=3, stride=1, padding=1)
        self.squeeze_excite1 = Squeeze_Excite_Block(filters[1])

        self.residual_conv_2 = ResidualConv(filters[1], filters[2], kernel_size=3, stride=1, padding=1)
        self.squeeze_excite2 = Squeeze_Excite_Block(filters[2])

        # self.bridge = ResidualConv(filters[2], filters[3], kernel_size=3, stride=1, padding=1)
        # self.squeeze_excite3 = Squeeze_Excite_Block(filters[3])

        self.aspp = ASPP(filters[2], filters[2])

        self.upsample_1 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[2] + filters[2], filters[1], kernel_size=3, stride=1, padding=1)

        self.upsample_2 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[1] + filters[1], filters[0], kernel_size=3, stride=1, padding=1)

        self.upsample_3 = Upsample(filters[0], filters[0], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[0] + filters[0], filters[0], kernel_size=3, stride=1, padding=1)
        self.last = nn.Conv2d(filters[0], 1, kernel_size=1, stride=1)

        self.iwt = InverseHaarTransform(4)

    def forward(self, x):
        # Encode

        # wavelet = self.dwt(x)

        # x1 = 0.1*self.input_layer(wavelet) + self.input_skip(wavelet)
        x = self.input_layer(x)+ self.input_skip(x)
        x1 = self.squeeze_excite0(x)
        x1_1 = self.Maxpool(x1) # (1,64,128,128)

        x2 = self.residual_conv_1(x1_1)
        x2 = self.squeeze_excite1(x2)
        x2_1 = self.Maxpool(x2) # (1,128,64,64)

        x3 = self.residual_conv_2(x2_1)
        x3 = self.squeeze_excite2(x3)
        x3_1 = self.Maxpool(x3) # (1,256,32,32)
        # Bridge

        #aspp
        x4 = self.aspp(x3_1) # (1,256,32,32)
        # Decode
        x4 = self.upsample_1(x4) # (1,256,64,64)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        out = self.last(x10)
        # output = self.iwt(out)

        return out

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, device = 'cuda'):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net    

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_sobel = SobelConv2d(in_ch, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.Conv1 = conv_block(in_ch, filters[0]) # 改了输入通道 1->64
        self.Conv2 = conv_block(filters[0]+32, filters[1]) # 96 128
        self.Conv3 = conv_block(filters[1]+32, filters[2]) # 160 256
        self.Conv4 = conv_block(filters[2]+32, filters[3]) # 288 512
        self.Conv5 = conv_block(filters[3]+32, filters[4]) # 544 1024

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        edge_1 = self.conv_sobel(x)
        # x = torch.cat((x, edge_1), dim=1)
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        edge_2 = self.Maxpool1(edge_1)
        e2 = self.Conv2(torch.cat((e2, edge_2), dim=1)) #多了一个32通道的边缘信息
        # e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        edge_3 = self.Maxpool2(edge_2)
        e3 = self.Conv3(torch.cat((e3, edge_3), dim=1)) #多了一个32通道的边缘信息
        # e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        edge_4 = self.Maxpool3(edge_3)
        e4 = self.Conv4(torch.cat((e4, edge_4), dim=1)) #多了一个32通道的边缘信息      
        # e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        edge_5 = self.Maxpool3(edge_4)
        e5 = self.Conv5(torch.cat((e5, edge_5), dim=1)) #多了一个32通道的边缘信息          
        # e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out

class SobelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2
        
        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        sobel_weight = self.sobel_weight * self.sobel_factor

        if torch.cuda.is_available():
            sobel_weight = sobel_weight.cuda()

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out


class EDCNN(nn.Module):

    def __init__(self, in_ch=1, out_ch=32, sobel_ch=32):
        super(EDCNN, self).__init__()

        self.conv_sobel = SobelConv2d(in_ch, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_p1 = nn.Conv2d(in_ch + sobel_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p2 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p3 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p4 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f4 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p5 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f5 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p6 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f6 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p7 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f7 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p8 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f8 = nn.Conv2d(out_ch, in_ch, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU()
    def forward(self, x):
        out_0 = self.conv_sobel(x)
        out_0 = torch.cat((x, out_0), dim=-3)

        out_1 = self.relu(self.conv_p1(out_0))
        out_1 = self.relu(self.conv_f1(out_1))
        out_1 = torch.cat((out_0, out_1), dim=-3)

        out_2 = self.relu(self.conv_p2(out_1))
        out_2 = self.relu(self.conv_f2(out_2))
        out_2 = torch.cat((out_0, out_2), dim=-3)

        out_3 = self.relu(self.conv_p3(out_2))
        out_3 = self.relu(self.conv_f3(out_3))
        out_3 = torch.cat((out_0, out_3), dim=-3)

        out_4 = self.relu(self.conv_p4(out_3))
        out_4 = self.relu(self.conv_f4(out_4))
        out_4 = torch.cat((out_0, out_4), dim=-3)

        out_5 = self.relu(self.conv_p5(out_4))
        out_5 = self.relu(self.conv_f5(out_5))
        out_5 = torch.cat((out_0, out_5), dim=-3)

        out_6 = self.relu(self.conv_p6(out_5))
        out_6 = self.relu(self.conv_f6(out_6))
        out_6 = torch.cat((out_0, out_6), dim=-3)

        out_7 = self.relu(self.conv_p7(out_6))
        out_7 = self.relu(self.conv_f7(out_7))
        out_7 = torch.cat((out_0, out_7), dim=-3)

        out_8 = self.relu(self.conv_p8(out_7))
        out_8 = self.conv_f8(out_8)

        return out_8