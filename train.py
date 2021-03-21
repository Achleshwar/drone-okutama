import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.datautils import *
from utils.loss import *


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

class AverageMeter(object):
    """
    Computes the average value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class Timer(object):
    """
    class to do timekeeping
    """
    def __init__(self):
        self.last_time=time.time()
        
    def timeit(self):
        old_time=self.last_time
        self.last_time=time.time()
        return self.last_time-old_time

def train_okutama(data_loader, model, device, optimizer, epoch):
    
    actions_meter=AverageMeter()
    # activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    epoch_timer=Timer()

    #parameters
    B = 2
    T = 5
    num_boxes = 12
    for i, batch_data in enumerate(data_loader):

        print("==> Loading sample no. : ", i * B * T)
        model.train()
        model.apply(set_bn_eval)
    
        # prepare batch data
        batch_data=[b.to(device=device) for b in batch_data]
        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]

        # forward
        actn, actions_scores=model((batch_data[0],batch_data[1],batch_data[3]))
        actions_scores = torch.reshape(actions_scores, (B*T,num_boxes)).to(device=device)
        # print(actions_scores.shape)

        # actions_scores = actions_scores.unsqueeze(0)
        # actions_scores = torch.zeros(actions_scores.size(0), 15).scatter_(1, actions_scores, 1.)
        
        actions_in=batch_data[2].reshape((batch_size,num_frames,num_boxes))
        # print(actions_in.shape)
        # activities_in=batch_data[3].reshape((batch_size,num_frames))
        bboxes_num=batch_data[3].reshape(batch_size,num_frames)

        actions_in_nopad=[]
        # if cfg.training_stage==1:
        actions_in=actions_in.reshape((batch_size*num_frames,num_boxes,))
        bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
        for bt in range(batch_size*num_frames):
            N=bboxes_num[bt]
            actions_in_nopad.append(actions_in[bt,:N])
        # else:
        #     for b in range(batch_size):
        #         N=bboxes_num[b][0]
        #         actions_in_nopad.append(actions_in[b][0][:N])
        # actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
        # if cfg.training_stage==1:
        #     activities_in=activities_in.reshape(-1,)
        # else:
        #     activities_in=activities_in[:,0].reshape(batch_size,)
        
        # Predict actions
        # print("shape of actions_scores = ", actions_scores.shape)
        # print("shape of actions_in = ", actions_in.shape)
        # actions_in = torch.reshape(actions_in, (B,T,num_boxes)).to(device=device)
        # print("actions_in = ", actions_in)
        # print("actions_scores = ",actions_scores)
        # actions_scores = Variable(actions_scores.float(), requires_grad = True)
        # actions_in = Variable(actions_in.float(), requires_grad = True)
        # loss = nn.MultiLabelMarginLoss()
        
        # actions_loss = loss(actions_scores.int(), actions_in)
        # actions_loss = Variable(actions_loss, requires_grad = True)
        # actions_loss=F.cross_entropy(actions_scores,actions_in,weight=None)  
        # actions_labels=torch.argmax(actions_scores,dim=1)  #B*T*N,
        # print("actions_labels = ",actions_labels)


        actions_loss = binary_cross_entropy(actn.view(1,actn.size(0), actn.size(1)), actions_in)
        actions_correct=torch.sum(torch.eq(actions_scores.int(),actions_in.int()).float())

        # # Predict activities
        # activities_loss=F.cross_entropy(activities_scores,activities_in)
        # activities_labels=torch.argmax(activities_scores,dim=1)  #B*T,
        # activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
        
        
        # Get accuracy
        actions_accuracy=actions_correct.item()/(actions_scores.shape[0] * num_boxes)
        # activities_accuracy=activities_correct.item()/activities_scores.shape[0]
        
        actions_meter.update(actions_accuracy, actions_scores.shape[0])
        # activities_meter.update(activities_accuracy, activities_scores.shape[0])

        # Total loss
        # total_loss=actions_loss
        loss_meter.update(actions_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        actions_loss.backward()
        optimizer.step()
    
    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'actions_acc':actions_meter.avg*100
    }
    
    return train_info
