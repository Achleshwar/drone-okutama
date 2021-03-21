import os
import random
import numpy as np
import torch
from utils.dataset import *
from utils.basenet import *
from utils.model import *
from utils.datautils import *
from train import *
from test import *

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 


def train_net(training_loader, validation_loader, epochs):
    """
    training
    """
    
    # Set random seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Set data position
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Build model and optimizer
    
    model=Basenet_okutama()
    # if cfg.use_multi_gpu:
    #     model=nn.DataParallel(model)

    model=model.to(device=device)
    
    model.train()
    model.apply(set_bn_eval)

    train_learning_rate = 1e-2  #initial learning rate 
    lr_plan = {41:1e-4, 81:5e-5, 121:1e-5}  #change learning rate in these epochs 
    train_dropout_prob = 0.3  #dropout probability
    weight_decay = 0
    
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=train_learning_rate,weight_decay=weight_decay)
    
    # if cfg.test_before_train:
    #     test_info=test(validation_loader, model, device, 0, cfg)
    #     print(test_info)

    # Training iteration
    best_result={'epoch':0, 'actions_acc':0}
    start_epoch=1
    max_epoch = epochs
    for epoch in range(start_epoch, start_epoch+max_epoch):
        print("Epoch number = ", epoch)
        if epoch in lr_plan:
            adjust_lr(optimizer, lr_plan[epoch])
            
        # One epoch of forward and backward
        train_info=train_okutama(training_loader, model, device, optimizer, epoch)
        # show_epoch_info('Train', cfg.log_path, train_info)
        print(train_info)


        """
        TODO: debug error in `test_okutama()`
        """
        # # Test
        # test_interval_epoch = 2

        # if epoch % test_interval_epoch == 0:
        #     test_info=test_okutama(validation_loader, model, device, epoch)
        #     # show_epoch_info('Test', cfg.log_path, test_info)
        #     print(test_info)
            
        #     if test_info['actions_acc']>best_result['actions_acc']:
        #         best_result=test_info
        #     # print_log(cfg.log_path, 
        #     #           'Best group activity accuracy: %.2f%% at epoch #%d.'%(best_result['activities_acc'], best_result['epoch']))
        #     print('Best accuracy: %.2f%% at epoch #%d.'%(best_result['actions_acc'], best_result['epoch']))
            
        #     # Save model
        #     if cfg.training_stage==2:
        #         state = {
        #             'epoch': epoch,
        #             'state_dict': model.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #         }
        #         filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['activities_acc'])
        #         torch.save(state, filepath)
        #         print('model saved to:',filepath)   
        #     elif cfg.training_stage==1:
        #     for m in model.modules():
        #         if isinstance(m, Basenet):

        #             filepath=cfg.result_path+'/epoch%d_%.2f%%.pth'%(epoch,test_info['actions_acc'])
        #             m.savemodel(filepath)
        #             print('model saved to:',filepath)
        #     else:
        #         assert False