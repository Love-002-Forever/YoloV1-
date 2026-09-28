import torch
from torch import nn


class Loss(nn.Module):
    def __init__(self,num_boxes=2,num_classes=4,lambda_coord=5,lambda_noobj=0.5):
        # lambda_coord 是 YOLO 损失函数中用于加权边界框坐标误差的超参数，YOLOv1 中默认取 5
        # lambda_noobj 是 YOLO v1 损失函数中用来平衡“无目标网格”置信度损失的权重，默认值通常取 0.5‌。它的作用是降低背景网格对损失的影响，防止模型过度偏向预测“没有物体”
        super(Loss, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')


    def forward(self, pred,target):
        # pred: (B,C,7,7)
        # target: (B,C,7,7)
        # 需reshape为(B, 7, 7, num_boxes, 5 + num_classes)

        B = pred.shape[0]
        pred = pred.permute(0,2,3,1).view(B,7,7,self.num_boxes, 5 + self.num_classes)
        #print(pred.shape) # pred = torch.randn(2,18,7,7) --> torch.Size([2, 7, 7, 2, 9])
        target = target.permute(0,2,3,1).view(B,7,7,self.num_boxes, 5 + self.num_classes)


        box1_input = self.IoU(pred[...,0,:4],target[...,0,:4]) # (B,7,7,num_classes)
        box2_input = self.IoU(pred[...,1,:4],target[...,0,:4])

        # IoU 正负样本
        has_obj = target[...,0,4] > 0
        box1_better = (box1_input >= box2_input) & has_obj
        box2_better = (box2_input > box1_input) & has_obj

        # 拼出 0/1 掩码
        obj_mask = torch.zeros(B,7,7,self.num_boxes,device=pred.device)

        obj_mask[...,0] = box1_better.float()
        obj_mask[...,1] = box2_better.float()
        #obj_mask = obj_mask.unsqueeze(-1) # 在末尾插入维度为1 (B,7,7,2) -> (B,7,7,2,1)

        noobj_mask = 1 - obj_mask # 负样本 1->0; 0->1
        # 需要扩展 obj_mask 以匹配 xy (..., 2) 和 wh (..., 2) 的最后维度
        # obj_mask: [B, 7, 7, 2] -> [B, 7, 7, 2, 1]
        mask_coord = obj_mask.unsqueeze(-1)

        l1 = (self.mse(pred[...,:2] * mask_coord,target[...,:2] * mask_coord) +
              self.mse(torch.sqrt(torch.clamp(pred[...,2:4],min=1e-8)) * mask_coord,torch.sqrt(torch.clamp(target[...,2:4],min=1e-8)) * mask_coord))
        l2 = (self.mse(pred[...,4],target[...,4] * obj_mask))
        l3 = (self.mse(pred[...,4],target[...,4] * noobj_mask))
        l4 = (self.mse(pred[...,-self.num_classes:] * (mask_coord > 0),target[...,-self.num_classes:] * (mask_coord > 0)).float())

        total_loss = (
            self.lambda_coord * l1 +
            l2 +
            self.lambda_noobj * l3 +
            l4
        )

        return total_loss



    # IoU 最常见的意思是‌交并比（Intersection over Union）‌，是目标检测和图像分割里衡量预测框和真实框重叠程度的指标
    def IoU(self,box1,box2):
        # cx,cy,w,h
        # 1. 中心宽高 → 左上右下
        # 框1
        b1_cx1 = box1[..., 0] - box1[..., 2] / 2
        b1_cy1 = box1[..., 1] - box1[..., 3] / 2
        b1_cx2 = box1[...,0] +  box1[..., 2] / 2
        b1_cy2 = box1[...,1] + box1[..., 3] / 2
        # 框2
        b2_cx1 = box2[..., 0] - box2[..., 2] / 2
        b2_cy1 = box2[..., 1] - box2[..., 3] / 2
        b2_cx2 = box2[...,0] + box2[..., 2] / 2
        b2_cy2 = box2[...,1] + box2[..., 3] / 2

        # 2. 交集的左上角（取大）和右下角（取小）
        inter_x1 = torch.max(b1_cx1, b2_cx1)
        inter_y1 = torch.max(b1_cy1,b2_cy1)
        inter_x2 = torch.min(b1_cx2, b2_cx2)
        inter_y2 = torch.min(b1_cy2, b2_cy2)

        # 交集宽高，无交集为0
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h # 交集面积

        # 并集面积
        b1_area = box1[...,2] * box1[...,3]
        b2_area = box2[...,2] * box2[...,3]
        union_area = b1_area + b2_area - inter_area

        IoU = inter_area / (union_area + 1e-10)
        return IoU




if __name__ == '__main__':
    loss = Loss()
    pred = torch.randn(2,18,7,7)
    target = torch.randn(2,18,7,7)
    print(loss(pred,target).size())
    print(f"Total Loss Shape: {loss(pred,target).shape}")