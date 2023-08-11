import random
import os
import numpy as np
import pandas as pd
import torch
from cleanlab.filter import find_label_issues
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score
from thop import profile
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.patheffects as PathEffects
from typing import List
import mat73
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


'''设置随机种子'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


'''归一化与训练数据'''
def Data_prepared(HP):
    X_train, X_val, X_test, value_Y_train, value_Y_val, Y_test = HP.dataset
    '''归一化'''
    min_value = X_train.min()
    min_in_val = X_val.min()
    if min_in_val < min_value:
        min_value = min_in_val
    max_value = X_train.max()
    max_in_val = X_val.max()
    if max_in_val > max_value:
        max_value = max_in_val

    X_train = (X_train - min_value) / (max_value - min_value)
    X_val = (X_val - min_value) / (max_value - min_value)
    X_test = (X_test - min_value) / (max_value - min_value)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(value_Y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=HP.batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(value_Y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=HP.batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=HP.batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader

'''训练模型过程'''
def train(model,HP,train_dataloader):
    model.train()
    model = model.to(HP.device)
    correct = 0
    classifier_loss = 0
    r1_loss = 0
    result_loss = 0
    # 添加优化器
    optimizer1 = HP.optimizer1([{'params': model.parameters()}], lr=HP.lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = HP.optimizer2([{'params': model.lamda}], alpha=HP.alpha, device=HP.device, lr=HP.lr, momentum=0.9, weight_decay=0.0001)
    for data, target in train_dataloader:
        target = target.long()
        data = data.to(HP.device)
        target = target.to(HP.device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        output = model(data)
        # 交叉熵损失
        loss_fn = HP.loss.to(HP.device)
        classifier_loss_batch = loss_fn(output[1], target)
        # 正则化项-L1范数
        zero_data = torch.zeros(model.lamda.size())
        if torch.cuda.is_available():
            zero_data = zero_data.to(HP.device)
        r1_loss_batch = F.l1_loss(model.lamda, zero_data, reduction='sum')
        # 总loss
        result_loss_batch = classifier_loss_batch + HP.alpha * r1_loss_batch
        result_loss_batch.backward()
        optimizer1.step()
        optimizer2.step()
        classifier_loss += classifier_loss_batch.item()
        r1_loss += r1_loss_batch.item()
        result_loss += result_loss_batch.item()
        pred = output[1].argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    result_loss /= len(train_dataloader.dataset)
    return r1_loss, result_loss, correct

'''验证模型过程'''
def val(model,HP,val_dataloader):
    model.eval()
    model = model.to(HP.device)
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            target = target.long()
            data = data.to(HP.device)
            target = target.to(HP.device)
            output = model(data)
            loss_fn = HP.loss.to(HP.device)
            loss += loss_fn(output[1], target)
            pred = output[1].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(val_dataloader.dataset)
    return loss, correct

'''训练与验证模型过程'''
def train_and_val(model,HP_tr,HP_val,train_dataloader,val_dataloader):
    current_loss = 100
    for epoch in range(1, HP_tr.epoch + 1):
        #train
        r1_loss, train_loss, train_correct = train(model,HP_tr, train_dataloader)
        print('Train Epoch: {} \nTrain set: Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
            epoch,
            train_loss,
            train_correct,
            len(train_dataloader.dataset),
            100.0 * train_correct / len(train_dataloader.dataset))
        )
        writer = SummaryWriter(HP_tr.log_path)
        writer.add_scalar('Accuracy/train', 100.0 * train_correct / len(train_dataloader.dataset), epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('L1_Loss/train', r1_loss, epoch)
        #val
        val_loss, val_correct= val(model,HP_val,val_dataloader)
        fmt = 'Validation set: Loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
        print(
            fmt.format(
                val_loss,
                val_correct,
                len(val_dataloader.dataset),
                100.0 * val_correct / len(val_dataloader.dataset),
            )
        )
        writer = SummaryWriter(HP_val.log_path)
        writer.add_scalar('Accuracy/val', 100.0 * val_correct / len(val_dataloader.dataset), epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        #save best model
        if val_loss < current_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_loss, val_loss))
            current_loss = val_loss
            torch.save(model, HP_tr.save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")

'''
fuction: 修剪对应行
'''
def prune_new(input1,input2,HP):
    input1 = input1.cpu().detach_().numpy()
    input2 = input2.cpu().detach_().numpy()
    i = np.nonzero(input1)
    input2 = np.array(input2)
    input2_new = input2[i, :]
    input2_new_ = torch.tensor(input2_new, dtype=torch.float32)
    input2_new_ = input2_new_.to(HP.device)
    input2_new_ = input2_new_.squeeze()
    return input2_new_
'''
fuction: 修剪对应列
'''
def prune_new2(input1,input2,HP):
    input1 = input1.cpu().detach_().numpy()
    input2 = input2.cpu().detach_().numpy()
    i = np.nonzero(input1)
    input2 = np.array(input2)
    input2_new = input2[:, i]
    input2_new_ = torch.tensor(input2_new, dtype=torch.float32)
    input2_new_ = input2_new_.to(HP.device)
    input2_new_ = input2_new_.squeeze()
    return input2_new_
'''
fuction: 修剪对应元素
'''
def prune_new3(input1,input2,HP):
    input1 = input1.cpu().detach_().numpy()
    input2 = input2.cpu().detach_().numpy()
    i = np.nonzero(input1)
    input2 = np.array(input2)
    input2_new = input2[i]
    input2_new_ = torch.tensor(input2_new, dtype=torch.float32)
    input2_new_ = input2_new_.to(HP.device)
    input2_new_ = input2_new_.squeeze()
    return input2_new_

'''
fuction: 模型剪枝并计算相应指标
'''
def prune_model_and_test(model_prune,num,HP,test_dataloader):
    # savepath已保存的模型
    # loadpath剪枝后模型
    load_path = HP.load_path
    save_path = HP.save_path
    model = torch.load(load_path)
    torch.save(model.state_dict(), save_path)
    dict = torch.load(save_path)
    # 进行剪枝
    tensor_new = prune_new(dict["lamda"], dict["linear1.weight"], HP)
    dict["linear1.weight"] = tensor_new
    tensor_new2 = prune_new2(dict["lamda"], dict["linear2.weight"], HP)
    dict["linear2.weight"] = tensor_new2
    tensor_new3 = prune_new3(dict["lamda"], dict["linear1.bias"], HP)
    dict["linear1.bias"] = tensor_new3
    tensor_lamda = prune_new3(dict["lamda"], dict["lamda"], HP)
    dict["lamda"] = tensor_lamda
    # 计算剪枝后linear层的计算量与参数量
    params_linear = dict["linear1.weight"].size()[0] * (dict["linear1.weight"].size()[1] + 1) + \
                    dict["linear2.weight"].size()[0] * (dict["linear2.weight"].size()[1] + 1)
    # flops_linear = 2*model.state_dict()["linear1.weight"].size()[0]*model.state_dict()["linear1.weight"].size()[1]+2*model.state_dict()["linear2.weight"].size()[0]*model.state_dict()["linear2.weight"].size()[1]
    flops_linear = dict["linear1.weight"].size()[0] * (dict["linear1.weight"].size()[1] + 1) + \
                   dict["linear2.weight"].size()[0] * (dict["linear2.weight"].size()[1] + 1)
    # 计算特征稀疏度
    m = 0
    for i in tensor_lamda.cpu().numpy():
        if i != 0:
            m = m + 1
    print('特征维度:', str(m))
    print('特征稀疏度:', str(m / 1024))
    # 保存剪枝后的新模型
    model_new = model_prune
    model_new.load_state_dict(dict)
    model_new = model_new.to(HP.device)
    torch.save(model_new, save_path)
    # 计算新模型的参数量与计算量
    input = torch.randn((1, 2, 6000))
    flops, params = profile(model_new.cpu(), inputs=(input,))
    print('flops:', str((flops + flops_linear) / 1000 ** 3) + " " + 'G' )
    print('params:',str((params + params_linear) / 1000 ** 2) + " " + 'M')
    # 测试相关指标
    model_new = model_new.to(HP.device)
    model_new.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            data = data.to(HP.device)
            target = target.to(HP.device)
            output = model_new(data)
            pred = output[1].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    fmt = '\nTest set: Accuracy: {}/{} ({:.6f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

