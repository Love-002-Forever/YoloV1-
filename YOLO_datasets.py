import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class YOLODatasets(Dataset):
    def __init__(self, img_folder, label_folder, transform=None, label_transform=None):
        self.images_folder = os.path.abspath(img_folder)
        self.labels_folder = os.path.abspath(label_folder)
        self.images_lst = os.listdir(self.images_folder)
        self.labels_lst = os.listdir(self.labels_folder)
        self.transform = transform
        self.label_transform = label_transform
        with open(os.path.dirname(self.images_folder) + '/classes.txt','r',encoding='UTF-8') as f:
            self.classes = f.read().splitlines()
        #print('classes:', self.classes)
        #print(self.images_folder, self.labels_folder)

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        image_name = self.images_lst[idx]
        image_txt_name = os.path.splitext(image_name)[0]
        image = Image.open(os.path.join(self.images_folder,image_name)).convert('RGB')
        with open(os.path.join(self.labels_folder,image_txt_name + '.txt'), 'r',encoding='UTF-8') as f:
            labels_lst = []
            labels = f.readlines() # [class_id x_center y_center width height]
            for label in labels:
                label = list(map(float,(label.strip('\n').split())))
                label[0] = int(label[0])
                # label[1] = label[1] * 7 - int(label[1] * 7)
                # label[2] = label[2] * 7 - int(label[2] * 7)
                labels_lst.append(label)

        if self.transform:
            image = self.transform(image)
        targets = self.encode_labels(labels_lst)
        return image_name,image, targets

    def encode_labels(self, boxes):
        grid_size = 7
        num_boxes = 2
        num_classes = len(self.classes)
        channel = num_boxes * (5 + num_classes)

        target = torch.zeros(channel, grid_size, grid_size)

        for cls_id,cx,cy,w,h in boxes:
            # 网格位置
            grid_x = int(cx * grid_size)
            grid_y = int(cy * grid_size)
            # 防越界
            grid_x = min(grid_x, grid_size-1)
            grid_y = min(grid_y, grid_size-1)

            # 网格内置偏移
            dx = cx * grid_size - grid_x
            dy = cy * grid_size - grid_y

            # 填到对应网格位置
            target[0, grid_y, grid_x] = dx  # x偏移
            target[1, grid_y, grid_x] = dy  # y偏移
            target[2, grid_y, grid_x] = w  # 宽高直接用归一化值
            target[3, grid_y, grid_x] = h
            target[4, grid_y, grid_x] = 1.0  # 置信度=1

            target[num_boxes * 5 + cls_id, grid_y, grid_x] = 1.0

        return target



if __name__ == '__main__':
    yolo_dataset = YOLODatasets('./HelmetDataset-YOLO-Train/images', './HelmetDataset-YOLO-Train/labels',
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((448,448))]),
                                label_transform=None)
    with open('./HelmetDataset-YOLO-Train/classes.txt','r') as f:
        classes = f.read().splitlines()
        print(classes)
    image,target = yolo_dataset[0]
    print(target.shape)
    # for lb in target:
    #     classifier = int(lb[0])
    #     center_x = lb[1]
    #     center_y = lb[2]
    #     w = lb[3]
    #     h = lb[4]
    #     print(classifier,center_x,center_y,w,h)