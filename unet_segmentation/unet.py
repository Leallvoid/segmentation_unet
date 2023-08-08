import torch
import torch.nn as nn
from torch.nn import functional as F
from do_conv import DOConv2d
from aspp import _ASPP
from convlstm import ConvLSTM

# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.ReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)

class DOConv(nn.Module):
    def __init__(self, C_in, C_out):
        super(DOConv, self).__init__()
        self.layer = nn.Sequential(
            
            DOConv2d(C_in, C_out, 3, 9, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.ReLU(),
            
            DOConv2d(C_out, C_out, 3, 9, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layer(x)

# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.Down(x)

# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# 主干网络
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # 4次下采样
        self.C1 = Conv(1, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.aspp = _ASPP(512, 1024, [6, 12, 18, 24])

        # 4次上采样
        self.U1 = UpSampling(1024)
        self.C5 = DOConv(1024, 512)
        self.U2 = UpSampling(512)
        self.C6 = DOConv(512, 256)
        self.U3 = UpSampling(256)
        self.C7 = DOConv(256, 128)
        self.U4 = UpSampling(128)
        self.C8 = DOConv(128, 64)
        
        self.convlstm_R4 = ConvLSTM(512, 16, (3, 3), 1, True, True, False)
        self.convlstm_R3 = ConvLSTM(256, 16, (3, 3), 1, True, True, False)
        self.convlstm_R2 = ConvLSTM(128, 16, (3, 3), 1, True, True, False)
        
        
        self.Th = nn.Sigmoid()
        self.pred = DOConv2d(64, 3, 3, 9, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.aspp(self.D4(R4))
        
        # 拼接部分
        input_R4 = R4.unsqueeze(1).expand(-1, 10, -1, -1, -1)
        self.convlstm_R4(input_R4)
        
        input_R3 = R3.unsqueeze(1).expand(-1, 10, -1, -1, -1)
        self.convlstm_R3(input_R3)
        
        input_R2 = R2.unsqueeze(1).expand(-1, 10, -1, -1, -1)
        self.convlstm_R2(input_R2)
        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C5(self.U1(Y1, R4))
        O2 = self.C6(self.U2(O1, R3))
        O3 = self.C7(self.U3(O2, R2))
        O4 = self.C8(self.U4(O3, R1))

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.Th(self.pred(O4))
