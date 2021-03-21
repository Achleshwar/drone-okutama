import torch
import torch.nn as nn

def binary_cross_entropy(predictions, target):
    '''
    predictions: (batch * num_frames, nboxes, num_class) 
                    # (10, variable (max - 12), 13)
    target: (batch * num_frames, nboxes)
    '''

    if torch.cuda.is_available():
        predictions = predictions.to(device = "cuda")
        target = target.to(device = "cuda")

    else:
        predictions = predictions.to(device = "cpu")
        target = target.to(device = "cpu")

    bce_criterion = nn.BCEWithLogitsLoss()
    loss = 0
    for bt in range(predictions.size(0)):
        for nbox in range(predictions.size(1)):
            if nbox < 12:
                target_onehot = nn.functional.one_hot(torch.tensor(target[bt][nbox].int().item()), num_classes=13)
                loss += bce_criterion(predictions[bt][nbox].float(), target_onehot.float())
    loss = loss / (predictions.size(0) * predictions.size(1))
    
    return loss 