import torch
import torch.nn as nn
import torch.optim

import sys
import os

from Dataloader import Dataloader
from Models import MLP
from Losses import VariantInnerProduct
from Tools import ModelSaver


class TrainingModel(nn.Module):
    def __init__(self, inference_model, inner_product):
        super(TrainingModel, self).__init__()
        self.inference_model = inference_model
        self.inner_product = inner_product
    def forward(self, x, label):
        features = self.inference_model(x)
        # logits = self.inner_product(features, label)
        logits = self.inner_product(features)
        return features, logits
    def SaveInferenceModel():
        # TO BE DOWN
        return 0


# def Test(test_loder, model):
#     correct = 0
#     total = 0
#     for i, (data, target) in enumerate(test_loder):
#         if torch.cuda.is_available():
#             data = data.cuda()
#             target = target.cuda()

#         feats, logits = model(data, target)
#         _, predicted = torch.max(logits.data, 1)
#         total += target.size(0)
#         correct += (predicted == target.data).sum()
#     acc = (100. * float(correct)) / float(total)
#     print('Test Accuracy on the {} test images:{}/{} ({:.2f}%) \n' .format(total, correct, total, acc))
#     return acc


def Train(train_loader, model, criterion, optimizer, epoch, info_interval):
    for i, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        feats, logits = model(data, target)
        loss = criterion[0](logits, target.view(-1))

        _, predicted = torch.max(logits.data, 1)

        accuracy = (target.data == predicted).float().mean()

        optimizer[0].zero_grad()
        loss.backward()
        optimizer[0].step()

        if (i + 1) % info_interval == 0:
            print('Epoch [%d], Iter [%d/%d] Loss: %.4f Acc %.4f'
                  % (epoch, i + 1, len(train_loader) , loss.item(), accuracy))
    

def Processing(NumEpoch, LrScheduler, Optimizer, train_loader, test_loder, model, criterion, info_interval, save_path):
    # cur_best=0
    for epoch in range(NumEpoch):
        LrScheduler.step()
        print('Current Learning Rate: {}'.format(Optimizer.param_groups[0]['lr']))
        Train(train_loader, model, criterion, [Optimizer], epoch, info_interval)
        SavePath = save_path + str(epoch + 1) + '.model'
        ModelSaver.SaveModel(model, SavePath, epoch, 10)
        # cur_acc = Test(test_loder, model)
        # if cur_best < cur_acc:
        #     cur_best = cur_acc
        # print('Current best test accuracy is {:.2f}% \n'.format(cur_best))



def main():
    ################################################################################################
    # This process, set up the whole models and parameters
    # Get Hyper parameters and sets

    # General arg
    arg_DeviceIds = [0]
    arg_NumEpoch = 30
    arg_InfoInterval = 10
    arg_SavePath = './checkpoints/softmax_MYO_'
    arg_SaveEpochInterbal = 5

    # Data arg
    ###########################################
    arg_TrainDataPath = './data/data/data1/1'
    ###########################################
    arg_TrainBatchSize = 512
    arg_TestBatchSize = 1024

    arg_InputDim = 10

    arg_FeatureDim = 128
    arg_classNum = 53
    
    # Learning rate arg
    arg_BaseLr = 0.001
    arg_Momentum = 0.5
    arg_WeightDecay = 0.0005

    # Learning rate scheduler
    arg_LrEpochStep = 10
    arg_Gamma = 0.5

    # Dataset Loading
    TrainLoader, TestLoader = Dataloader.LoadCSV(arg_TrainBatchSize, arg_TestBatchSize, arg_TrainDataPath, isbalanced=False)
    

    # Model Constructing
    # Inference Model Constructing
    Inference = MLP(input_dim=arg_InputDim, feature_dim=arg_FeatureDim, dense_dim=16)
    # Innerproduct Construction
    # InnerProduct = torch.nn.Linear(arg_FeatureDim, arg_classNum)
    InnerProduct = VariantInnerProduct.NormalizedInnerProductWithScale(arg_FeatureDim, arg_classNum, scale=7.0)
    
    Model = torch.nn.DataParallel(TrainingModel(Inference, InnerProduct), arg_DeviceIds)

    # Losses and optimizers Defining
    # Softmax CrossEntropy
    SoftmaxLoss = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        SoftmaxLoss = SoftmaxLoss.cuda()
        Model = Model.cuda()
    criterion = [SoftmaxLoss]
    # Optimzer
    Optimizer = torch.optim.SGD(Model.parameters(), lr=arg_BaseLr, momentum=arg_Momentum, weight_decay=arg_WeightDecay)
    
    # Learning rate Schedule
    LrScheduler = torch.optim.lr_scheduler.StepLR(Optimizer, arg_LrEpochStep, gamma=arg_Gamma)


    ################################################################################################

    # Resume from a checkpoint/pertrain

    # Training models
    # Testing models
    Processing(arg_NumEpoch, LrScheduler, Optimizer, TrainLoader, TestLoader, Model, criterion, arg_InfoInterval, arg_SavePath)



if __name__ == '__main__':
    main()
