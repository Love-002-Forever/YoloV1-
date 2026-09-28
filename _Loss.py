from torch import nn

class LossAcc(nn.Module):
    def __init__(self):
        super(LossAcc, self).__init__()
        self.MSE_loss = nn.MSELoss() # MSE（Mean Squared Error，均方误差）用于衡量模型预测值与真实值之间的整体差异。
        self.CE_loss = nn.CrossEntropyLoss() # 用来衡量模型预测和真实标签差多少的损失函数‌，值越小说明预测越准。

    def forward(self,predicts,targets):
        predict_coordinate = predicts[:,:4]
        predict_classes = predicts[:,4:]
        targets_coordinate = targets[:,:4]
        targets_classes = targets[:,4:]
        return self.MSE_loss(predict_coordinate,targets_coordinate) + self.CE_loss(predict_classes,targets_classes)


if __name__=='__main__':
    import torch
    from torchvision import transforms
    from YOLO_datasets import YOLODatasets
    yolo_dataset = YOLODatasets('./HelmetDataset-YOLO-Train/images', './HelmetDataset-YOLO-Train/labels',
                                transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((448, 448))]),
                                label_transform=None)
    loss = LossAcc()
    image,target = loss(yolo_dataset[0])
    print(target)