from torch.autograd import Variable

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train_okutama(data_loader, model, device, optimizer, epoch):
    
    actions_meter=AverageMeter()
    # activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    epoch_timer=Timer()

    #parameters
    B = 2
    T = 5
    num_boxes = 12
    for batch_data in data_loader:
        model.train()
        model.apply(set_bn_eval)
    
        # prepare batch data
        batch_data=[b.to(device=device) for b in batch_data]
        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]

        # forward
        actions_scores=model((batch_data[0],batch_data[1],batch_data[3]))
        actions_scores = torch.reshape(actions_scores, (B*T,num_boxes)).to(device=device)
        # print(actions_scores.shape)

        # actions_scores = actions_scores.unsqueeze(0)
        # actions_scores = torch.zeros(actions_scores.size(0), 15).scatter_(1, actions_scores, 1.)
        
        actions_in=batch_data[2].reshape((batch_size,num_frames,num_boxes))
        bboxes_num=batch_data[3].reshape(batch_size,num_frames)

        actions_in_nopad=[]
        actions_in=actions_in.reshape((batch_size*num_frames,num_boxes,))
        bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
        for bt in range(batch_size*num_frames):
            N=bboxes_num[bt]
            actions_in_nopad.append(actions_in[bt,:N])

        # Predict actions
        # print("shape of actions_scores = ", actions_scores.shape)
        # print("shape of actions_in = ", actions_in.shape)
        # actions_in = torch.reshape(actions_in, (B,T,num_boxes)).to(device=device)
        # print("actions_in = ", actions_in)
        # print("actions_scores = ",actions_scores)
        # actions_scores = Variable(actions_scores.float(), requires_grad = True)
        # actions_in = Variable(actions_in.float(), requires_grad = True)
        loss = nn.MultiLabelMarginLoss()
        
        actions_loss = loss(actions_scores, actions_in)
        actions_loss = Variable(actions_loss, requires_grad = True)
        # actions_loss=F.cross_entropy(actions_scores,actions_in,weight=None)  
        # actions_labels=torch.argmax(actions_scores,dim=1)  #B*T*N,
        # print("actions_labels = ",actions_labels)
        actions_correct=torch.sum(torch.eq(actions_scores.int(),actions_in.int()).float())
        
        
        # Get accuracy
        actions_accuracy=actions_correct.item()/(actions_scores.shape[0] * num_boxes)
        
        actions_meter.update(actions_accuracy, actions_scores.shape[0])

        # Total loss
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


