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


def test_okutama(data_loader, model, device, epoch):
    model.eval()
    
    actions_meter=AverageMeter()
    # activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    num_boxes = 12
    B = 2
    T = 5
    epoch_timer=Timer()
    with torch.no_grad():
        for batch_data in data_loader:
            # prepare batch data
            batch_data=[b.to(device=device) for b in batch_data]
            batch_size=batch_data[0].shape[0]
            num_frames=batch_data[0].shape[1]
            
            actions_in=batch_data[2].reshape((batch_size,num_frames, num_boxes))
            # activities_in=batch_data[3].reshape((batch_size,num_frames))
            bboxes_num=batch_data[3].reshape(batch_size,num_frames)

            # forward
            actions_scores=model((batch_data[0],batch_data[1],batch_data[3]))
            actions_scores = torch.reshape(actions_scores, (B*T,num_boxes)).to(device=device)
            actions_in_nopad=[]
            

            actions_in=actions_in.reshape((batch_size*num_frames,num_boxes,))
            bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
            for bt in range(batch_size*num_frames):
                N=bboxes_num[bt]
                actions_in_nopad.append(actions_in[bt,:N])

            loss = nn.MultiLabelMarginLoss()
            actions_loss = loss(actions_scores, actions_in)
            actions_loss = Variable(actions_loss, requires_grad = True)


            actions_correct=torch.sum(torch.eq(actions_scores.int(),actions_in.int()).float())

            
            # Get accuracy
            actions_accuracy=actions_correct.item()/(actions_scores.shape[0] * num_boxes)

            actions_meter.update(actions_accuracy, actions_scores.shape[0])

            # Total lossloss_meter.update(actions_loss.item(), batch_size)

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'actions_acc':actions_meter.avg*100
    }

    return test_info