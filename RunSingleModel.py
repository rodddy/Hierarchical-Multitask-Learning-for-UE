import math

import torch
import torch.nn as nn
import torch.utils.data as Data
from datetime import datetime

from gensim.downloader import base_dir
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,roc_auc_score
from constantly import Flags
from sqlalchemy.sql.operators import truediv
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
from collections import Counter
import torch.nn.functional as F
import Models.CNNLSTMSTFusion as CNNLSTMFusion
from keras.integration_test.preprocessing_test_utils import BATCH_SIZE

#LR = 0.001
LR = 0.005
Epoch = 80
base_dir = os.path.dirname(os.getcwd())
print(base_dir)
#load data
def load_data(IndexOfAction = 5):
    train_data = np.load(os.path.join(base_dir,f'dataset/subtask/{IndexOfAction}/x_train_4.npy'))
    train_labels = np.load(os.path.join(base_dir,f'dataset/subtask/{IndexOfAction}/y_train_4.npy'))
    val_data = np.load(os.path.join(base_dir,f'dataset/subtask/{IndexOfAction}/x_test_4.npy'))
    val_labels = np.load(os.path.join(base_dir,f'dataset/subtask/{IndexOfAction}/y_test_4.npy'))
    return train_data, val_data, train_labels, val_labels

def load_dataset(IndexOfAction = 5):

    train_data, val_data, train_labels, val_labels = load_data(IndexOfAction)
    Xtrain_tensor = torch.tensor(train_data, dtype=torch.float32)
    print(Xtrain_tensor.shape)
    Ytrain_tensor = torch.tensor(train_labels, dtype=torch.long)  # 标签从 1 开始，需要减 1
    Xval_tensor = torch.tensor(val_data, dtype=torch.float32)
    Yval_tensor = torch.tensor(val_labels, dtype=torch.long)  # 标签从 1 开始，需要减 1
    train_dataset = TensorDataset(Xtrain_tensor, Ytrain_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(Xval_tensor, Yval_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader
#5,7,11,13,15
#[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
num_classes = 3
ActionIndex = 17
FrozenLayers = 0

train_loader, test_loader = load_dataset(ActionIndex)

def adjust_learning_rate(optimizer, epoch):
    lr = LR * (0.1 ** (epoch // 50))

    optimizer.param_groups[0]['lr'] = lr
epoch = 0
# model = ResNet(BasicBlock, [2,2,2,2], num_classes)
model = CNNLSTMFusion.CNNLSTMSTFusion(input_channels=18, hidden_dim=64, kernel_size=5, num_layers=2,
                                                num_classes=num_classes)
model.cpu()
#print(model)

strDir =  datetime.now().strftime("%Y%m%d%H%M%S")
writer = None
#writer = {
#    'train_loss': SummaryWriter(log_dir=f'runs/SingleTask/{ActionIndex}/{strDir}/train_loss'),
#    'train_acc': SummaryWriter(log_dir=f'runs/SingleTask/{ActionIndex}/{strDir}/train_acc'),
#    'val_loss': SummaryWriter(log_dir=f'runs/SingleTask/{ActionIndex}/{strDir}/val_loss'),
#    'val_acc': SummaryWriter(log_dir=f'runs/SingleTask/{ActionIndex}/{strDir}/val_acc'),

#}

loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
save_interval = 10

def train(epoch):
    adjust_learning_rate(optimizer, epoch)
    train_loss = 0
    train_num = 0
    model.train()
    for step, (x, y) in enumerate(train_loader):
        x, y = x.cpu(), y.cpu()
        output = model(x)
        loss = loss_f(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred = torch.max(output, 1)[1].cpu().numpy()
        label = y.cpu().numpy()
        train_num += (pred==label).sum()
    train_acc = train_num /  len(train_loader.dataset)
    train_loss_epoch = train_loss / len(train_loader)
    print('Train Epoch:{} Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch, train_loss/len(train_loader), train_acc),end='||')
    writer['train_acc'].add_scalar('Data', train_acc, epoch)
    writer['train_loss'].add_scalar('Data', train_loss_epoch, epoch)
    if (epoch + 1) % save_interval == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss_epoch,
        }, os.path.join('Models', 'checkpoints',
                        f'model_actionIndex_{ActionIndex}_epoch_{epoch + 1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'))
        print(f'Model saved at epoch {epoch + 1}')

def val(flag=False):
    test_loss = 0
    test_num = 0

    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(test_loader):
            x, y = x.cpu(), y.cpu()
            output = model(x)
            loss = loss_f(output, y)

            test_loss += loss.item()

            pred = torch.max(output, 1)[1].cpu().numpy()
            label = y.cpu().numpy()
            test_num += (pred==label).sum()
            all_preds.extend(pred)
            all_labels.extend(label)
    test_acc = test_num / len(test_loader.dataset)
    test_loss_epoch = test_loss / len(test_loader)
    writer['val_acc'].add_scalar('Data', test_acc, epoch)
    writer['val_loss'].add_scalar('Data', test_loss_epoch, epoch)

    print('Test Loss:{:.4f} Test Acc:{:.4f}'.format(test_loss/len(test_loader), test_acc))

def eval_model(flag=False):
    test_loss = 0
    test_num = 0

    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(test_loader):
            x, y = x.cpu(), y.cpu()
            output = model(x)
            loss = loss_f(output, y)

            test_loss += loss.item()

            pred = torch.max(output, 1)[1].cpu().numpy()
            label = y.cpu().numpy()
            test_num += (pred==label).sum()
            all_preds.extend(pred)
            all_labels.extend(label)
    test_acc = test_num / len(test_loader.dataset)
    test_loss_epoch = test_loss / len(test_loader)

    'print(Test Loss:{:.4f} Test Acc:{:.4f}'.format(test_loss/len(test_loader), test_acc)
    if flag==True:
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        #print(all_labels)
        np.save(os.path.join('Models', f'cm_{ActionIndex}.npy'), cm)

# 加载最新的模型检查点
checkpoint_path = 'Models/checkpoints'
for i in range(0,1):
    FrozenLayers = i
    strDir = datetime.now().strftime("%Y%m%d%H%M%S")
    writer = {
        'train_loss': SummaryWriter(log_dir=f'runs/SingleTask/{ActionIndex}/{strDir}/train_loss'),
        'train_acc': SummaryWriter(log_dir=f'runs/SingleTask/{ActionIndex}/{strDir}/train_acc'),
        'val_loss': SummaryWriter(log_dir=f'runs/SingleTask/{ActionIndex}/{strDir}/val_loss'),
        'val_acc': SummaryWriter(log_dir=f'runs/SingleTask/{ActionIndex}/{strDir}/val_acc'),

    }

    #modelname = os.path.join(checkpoint_path, 'model_actionIndex_5_0_epoch_80_20250305_204918.pth')
    #modelname = os.path.join(checkpoint_path, 'model_actionIndex_7_0_epoch_80_20250305_205246.pth')
    modelname = os.path.join(checkpoint_path, 'xxx.pth')

    if os.path.exists(modelname):
        checkpoint = torch.load(modelname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for name,child in model.named_children():
            if FrozenLayers==1:
                if name in ['conv1', 'relu1']:
                    for param in child.parameters():
                        param.requires_grad = False
            elif FrozenLayers==2:
                if name in ['conv1', 'relu1', 'conv2', 'relu2']:
                    for param in child.parameters():
                        param.requires_grad = False
            elif FrozenLayers==3:
                if name in ['conv1', 'relu1', 'conv2', 'relu2', 'conv3', 'relu3', 'se_block']:
                    for param in child.parameters():
                        param.requires_grad = False
            elif FrozenLayers==4:
                if name in ['conv1', 'relu1', 'conv2', 'relu2', 'conv3', 'relu3', 'conv5', 'relu5', 'se_block']:
                    for param in child.parameters():
                        param.requires_grad = False
            elif FrozenLayers==5:
                if name in ['conv1', 'relu1', 'conv2', 'relu2', 'conv3', 'relu3', 'conv5', 'relu5', 'se_block','lstm1']:
                    for param in child.parameters():
                        param.requires_grad = False
            elif FrozenLayers==6:
                if name in ['conv1', 'relu1', 'conv2', 'relu2', 'conv3', 'relu3', 'conv5', 'relu5', 'se_block','lstm1','lstm2']:
                    for param in child.parameters():
                        param.requires_grad = False

    for epoch in range(Epoch):
        train(epoch)
        val()
    eval_model(True)
