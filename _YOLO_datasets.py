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
        print('classes:', self.classes)
        #print(self.images_folder, self.labels_folder)

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.images_folder,self.images_lst[idx])).convert('RGB')
        with open(os.path.join(self.labels_folder,self.labels_lst[idx]), 'r',encoding='UTF-8') as f:
            labels_lst = []
            labels = f.readlines()
            for label in labels:
                label = list(map(float,(label.strip('\n').split())))
                labels_lst.append(label)

        if self.transform:
            image = self.transform(image)
        return image, torch.Tensor(labels_lst)



if __name__ == '__main__':
    yolo_dataset = YOLODatasets('./HelmetDataset-YOLO-Train/images', './HelmetDataset-YOLO-Train/labels',
                                transform=transforms.Compose([transforms.ToTensor()]),
                                label_transform=None)
    print(yolo_dataset[17])