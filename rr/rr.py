import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from monai.networks.nets import UNet
import random

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=3,
)
    
model_layer_map = {}
for name, module in model.named_modules():
    model_layer_map[name] = module
    
        
def get_model_layer(layer_name):
    return model_layer_map[layer_name]


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

replay_memory = []
for _ in range(5):
    x = torch.randn(15, 256, 10, 10)
    for c in x:
        replay_memory.append(c)


if __name__ == "__main__":
    start = time()
    print('-'*80)
    
    model_layer_name = 'model.1.submodule.1.submodule.1.submodule.1.submodule.conv.unit2.adn.A'
    model_layer = get_model_layer(model_layer_name)   
    model_layer.register_forward_hook(get_activation(model_layer_name))
    x = torch.randn(20, 1, 160, 160)
    y1 = model(x)
    latent_rep1 = activation[model_layer_name]    
    # print(f"Latent representation shape: {latent_rep.shape}")
    
    
    model_layer.register_forward_hook(get_activation(model_layer_name))
    # x = torch.randn(20, 1, 160, 160)
    y2 = model(x)
    latent_rep2 = activation[model_layer_name]    
    # print(f"Latent representation shape: {latent_rep.shape}")
    
    if torch.equal(latent_rep1, latent_rep2):
        print("Latent representations are equal")
        
    if torch.equal(y1, y2):
        print("Outputs are equal")
    
    # Get a same no of samples from replay memory
    replay_memory_samples = random.sample(replay_memory, x.shape[0])
    replay_memory_samples = torch.stack(replay_memory_samples)
    print(f"Replay memory samples shape: {replay_memory_samples.shape}")
    
    # Compute the loss between the latent representation and replay memory samples
    
    latent_rep_loss = F.l1_loss(latent_rep1, replay_memory_samples)
    print(f"Latent representation loss: {latent_rep_loss}")
    
    print('-'*80)
    print(f"Time taken: {time() - start :.2f} seconds")
    
    # model.1.submodule.1.submodule.1.submodule.1.residual
    # model.1.submodule.1.submodule.1.submodule.1.submodule.1.residual