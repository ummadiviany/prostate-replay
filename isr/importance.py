# imports and installs
import torch
from einops import rearrange
# ------------------------------------------------------------------------------------

def get_sample_importance(model, imgs, labels, criterion, optimizer):
    grad_sum = 0
    
    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
    labels = rearrange(labels, 'b c h w d -> (b d) c h w')
    
    # Set the gradients to zero
    optimizer.zero_grad()
    # Pass the images through the model
    preds = model(imgs)
    # Calculate the loss
    loss = criterion(preds, labels)
    # Backpropagate the loss
    loss.backward()
    
    for params in model.parameters():
        grad_sum += torch.sum(torch.abs(params.grad.data)).item()
    
    return grad_sum
