import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

    
class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w] 
        q = self.norm(q)
        return U * q 

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse
    

class generator(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(generator, self).__init__()                
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.c_se1 = scSE(64)        
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.c_se2 = scSE(128)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.c_se3 = scSE(256)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.c_se4 = scSE(512)
        self.conv5 = DoubleConv(512, 1024)
        self.c_se5 = scSE(1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.c_se6 = scSE(512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.c_se7 = scSE(256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.c_se8 = scSE(128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.c_se9 = scSE(64)
        self.conv10 = nn.Conv2d(64,out_ch, 1) 

    def forward(self,x):
        c1=self.conv1(x)
        c1=self.c_se1(c1)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        c2=self.c_se2(c2)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        c3=self.c_se3(c3)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        c4=self.c_se4(c4)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        c5=self.c_se5(c5)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        c6=self.c_se6(c6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        c7=self.c_se7(c7) 
        up_8=self.up8(c7)        
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        c8=self.c_se8(c8)
        up_9=self.up9(c8)       
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c9=self.c_se9(c9)
        c10=self.conv10(c9)
        #out = nn.Sigmoid()(c10)
        return c10

# class generator(nn.Module):
#     def __init__(self,in_ch,out_ch):
#         super(generator, self).__init__()
                
#         self.conv1 = DoubleConv(in_ch, 64)
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = DoubleConv(64, 128)
#         self.pool2 = nn.MaxPool2d(2)
#         self.conv3 = DoubleConv(128, 256)
#         self.pool3 = nn.MaxPool2d(2)
#         self.conv4 = DoubleConv(256, 512)
#         self.pool4 = nn.MaxPool2d(2)
#         self.conv5 = DoubleConv(512, 1024)
#         self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
#         self.conv6 = DoubleConv(1024, 512)
#         self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.conv7 = DoubleConv(512, 256)
#         self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.conv8 = DoubleConv(256, 128)
#         self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.conv9 = DoubleConv(128, 64)
#         self.conv10 = nn.Conv2d(64,out_ch, 1)

#     def forward(self,x):
#         c1=self.conv1(x)
#         p1=self.pool1(c1)
#         c2=self.conv2(p1)
#         p2=self.pool2(c2)
#         c3=self.conv3(p2)
#         p3=self.pool3(c3)
#         c4=self.conv4(p3)
#         p4=self.pool4(c4)
#         c5=self.conv5(p4)
#         up_6= self.up6(c5)
#         merge6 = torch.cat([up_6, c4], dim=1)
#         c6=self.conv6(merge6)
#         up_7=self.up7(c6)
#         merge7 = torch.cat([up_7, c3], dim=1)
#         c7=self.conv7(merge7)
#         up_8=self.up8(c7)
#         merge8 = torch.cat([up_8, c2], dim=1)
#         c8=self.conv8(merge8)
#         up_9=self.up9(c8)
#         merge9=torch.cat([up_9,c1],dim=1)
#         c9=self.conv9(merge9)
#         c10=self.conv10(c9)
#         # out = nn.Sigmoid()(c10)
#         return c10
