from torch import nn
from torchvision.models import vgg16,VGG16_Weights

class YOLOModel(nn.Module):
    def __init__(self, num_box,num_classes):
        super(YOLOModel, self).__init__()
        self.num_classes = num_classes
        self.output_channels = num_box * (5 + num_classes)
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        # VGG16输入448×448时，backbone输出形状为 (Batch, 512, 14, 14)
        # 我们加两层卷积做特征调整：
        # - 把通道从512升到1024，增强特征表达能力
        # - 下采样一次到7×7，正好对应grid_size=7的网格划分
        self.neck = nn.Sequential(
            # 1×1卷积先降通道，减少计算量（对应之前讲的1×1卷积降维技巧）
            nn.Conv2d(512,256,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,inplace=True),

            # 3×3卷积提取空间特征
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,inplace=True),

            # 下采样到目标网格尺寸：14×14 →7×7
            nn.Conv2d(512,1024,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1,inplace=True),
        )

        self.head = nn.Conv2d(1024,self.output_channels,kernel_size=1)


    def forward(self,x):
        x = self.vgg(x)
        x = self.neck(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    import torch
    input = torch.randn(2, 3, 448, 448)
    yolo = YOLOModel(2,4)
    output = yolo(input)
    print(yolo)
    print(output.shape)
