import os
import torch
import torchvision
from torch import nn
from PIL import Image,ImageDraw

class Draw(nn.Module):
    def __init__(self,num_boxes,num_classes,image_size,image_path_name):
        super(Draw,self).__init__()
        self.nums_boxes = num_boxes
        self.num_classes = num_classes
        self.image_size = image_size
        self.image_path_name = image_path_name
        self.class_names = ["motor","no helmet","number","with helmet"]
        self.class_colors = ["red","green","blue","yellow"]
        self.grid_size = 7
        self.conf_threshold = 0.5
        self.iou_threshold = 0.4


    def forward(self,x):
        B, C, W, H= x.shape
        x = x.permute(0,2,3,1).view(B,self.grid_size,self.grid_size,self.nums_boxes,5 + self.num_classes)
        grid_x, grid_y = torch.meshgrid(torch.arange(self.grid_size),torch.arange(self.grid_size),indexing='ij')
        grid_x = grid_x.to(x.device)
        grid_y = grid_y.to(x.device)
        grid_x = grid_x.float().unsqueeze(0).unsqueeze(-1) # torch.Size([7, 7]) -> torch.Size([1, 7, 7, 1])
        grid_y = grid_y.float().unsqueeze(0).unsqueeze(-1)

        cx = (grid_x + torch.sigmoid(x[...,0])) / self.grid_size #torch.sigmoid(x[...,0]).shape : torch.Size([1, 7, 7, 2])
        cy = (grid_y + torch.sigmoid(x[...,1])) / self.grid_size
        # print(cx,cy)

        w,h = x[...,2],x[...,3]
        w = torch.sigmoid(w)
        h = torch.sigmoid(h)
        # print(w,h)

        # 置信度
        conf = x[...,4]
        conf = torch.sigmoid(conf)
        keep = conf > self.conf_threshold

        boxes = torch.stack([cx,cy,w,h],dim=-1)
        boxes = boxes[keep] # 留下的框: (M, 4)
        conf = conf[keep] # 留下的置信度: (M,)
        classes = x[...,5:][keep] # 留下的类别: (M,)

        temp_x1 = boxes[:,0] - boxes[:,2]/2
        temp_y1 = boxes[:,1] - boxes[:,3]/2
        temp_x2 = boxes[:,0] + boxes[:,2]/2
        temp_y2 = boxes[:,1] + boxes[:,3]/2


        keep_indices = torchvision.ops.nms(torch.stack([temp_x1,temp_y1,temp_x2,temp_y2],dim=-1),conf,iou_threshold=self.iou_threshold)
        final_boxes = boxes[keep_indices]
        final_conf = conf[keep_indices]
        final_classes = classes[keep_indices]

        image = Image.open(self.image_path_name).convert('RGB').resize((self.image_size,self.image_size))
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        for i in range(len(final_boxes)):
            # 1. 把 tensor 转成数字，再乘上图片尺寸（因为现在是归一化坐标 0~1）
            x1 = (final_boxes[i, 0] - final_boxes[i, 2] / 2).float().item() * self.image_size
            y1 = (final_boxes[i, 1] - final_boxes[i, 3] / 2).float().item() * self.image_size
            x2 = (final_boxes[i, 0] + final_boxes[i, 2] / 2).float().item() * self.image_size
            y2 = (final_boxes[i, 1] + final_boxes[i, 3] / 2).float().item() * self.image_size

            # 确认类别
            cls_id = final_classes[i].argmax().item()
            cls_name = self.class_names[cls_id]
            conf = final_conf[i].item()

            # 画框
            draw.rectangle([x1, y1, x2, y2], outline=self.class_colors[cls_id])

            # 4. 写字
            draw.text((x1, y1 - 10), f"{cls_name}: {conf:.2f}", fill=self.class_colors[cls_id])

            # 保存
            draw_image = draw_image.convert('RGB')  # 确保是 RGB 模式
            save_dir = './temp_images'
            filename = os.path.basename(self.image_path_name).strip()
            draw_image.save(os.path.join(save_dir, filename))
            #draw_image.show()

            # try:
            #     draw_image.save(os.path.join(save_dir, filename))
            #     print("保存成功:", os.path.join(save_dir, filename))
            # except Exception as e:
            #     print("保存失败:", filename, repr(filename), e)
            #     print("图片模式:", draw_image.mode)


if __name__ == '__main__':
    pred = torch.randn(1,18,7,7)
    draw = Draw(num_boxes=2,num_classes=4,image_size=448,image_path_name="./HelmetDataset-YOLO-Train/images/new1.jpg")
    input = draw(pred)
    print(input)