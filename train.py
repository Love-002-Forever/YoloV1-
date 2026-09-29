import os
import torch
from torch import nn
import YOLO_datasets
import Model,Loss
from to_images import Draw
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练设备\n{:=^20}".format(str(device)))
model = Model.YOLOModel(2,4)
model = model.to(device)

train_images_path = './HelmetDataset-YOLO-Train/images'
train_labels_path = './HelmetDataset-YOLO-Train/labels'
val_images_path = './HelmetDataset-YOLO-Val/images'
val_labels_path = './HelmetDataset-YOLO-Val/labels'

yolo_images_train = YOLO_datasets.YOLODatasets(train_images_path, train_labels_path,
                                               transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((448,448))]),
                                               label_transform=None)
yolo_images_val = YOLO_datasets.YOLODatasets(val_images_path, val_labels_path,
                                             transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((448,448))]),
                                             label_transform=None)

load_images_train = DataLoader(yolo_images_train, batch_size=10, shuffle=True)
load_images_val = DataLoader(yolo_images_val, batch_size=10, shuffle=False)

writer = SummaryWriter("./logs")
criterion = Loss.Loss(num_boxes=2,num_classes=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)# weight_decay 惩罚过大的权重
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)


best_loss = float('inf')
patience_counter = 30
no_improve_epoch = 0


total_train_steps = 0
total_val_steps = 0
for epoch in range(100):
    print('Epoch {}/{}'.format(epoch + 1, 100))

    model.train()
    for step, (images_names,images, labels) in enumerate(load_images_train):
        name = os.path.join(train_images_path,images_names[0])
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # print("-"*10)
        # print(outputs.shape)
        # train_pred = outputs[0:1]
        # train_image = images[0:1]
        # print(train_pred[:,9])
        # print(train_pred.shape)
        # print(train_pred)
        # print("-" * 10)

        if total_train_steps % 10 == 0:
            draw = Draw(2,4,448,name)
            draw(outputs)
            print("画框完成")
        loss = criterion(outputs, labels)
        if loss.dim() > 0:
            loss = loss.mean()
        print("trian_loss:",loss.item())
        # acc = (outputs.argmax(1) == labels.argmax(1))
        # print('acc:',acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # writer.add_scalar('Loss/train', loss, total_train_steps)
        total_train_steps += 1

    model.eval()
    with torch.no_grad():
        for step, (images_names,images, labels) in enumerate(load_images_val):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            val_loss = criterion(pred, labels)
            if val_loss.dim() > 0:
                val_loss = val_loss.mean()
            print("val_loss:",val_loss.item())
            # writer.add_scalar('Loss/val', val_loss, total_val_steps)
            total_val_steps += 1
        scheduler.step()

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            torch.save(model.state_dict(), "./model.pth")
            print("已保存/更换更好的模型")
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1
            if no_improve_epoch > patience_counter:
                print("触发早停 epoch:",epoch+1)
                break

writer.close()