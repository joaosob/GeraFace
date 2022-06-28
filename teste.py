import pickle
import functools
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_old_G():
    with open('./ffhq.pkl', 'rb') as f:
        old_G = pickle.load(f)['G_ema'].cpu()
    return old_G

def plot(syn_images): 
    syn_images = (syn_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
    plt.axis('off') 
    resized_image = Image.fromarray(syn_images,mode='RGB').resize((369,369)) 
    plt.imshow(resized_image)
    plt.savefig(f'./face',bbox_inches='tight', transparent=False, pad_inches=0)
    torch.cuda.empty_cache()

G = load_old_G()

z_samples = np.random.RandomState().randn(1, G.z_dim) # [1, 512]
w_samples = G.mapping(torch.from_numpy(z_samples).to(torch.device('cpu')), None) 
w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32) # [1, 1, 512]
w_opt = torch.tensor(w_samples, dtype=torch.float32, device=torch.device('cpu'), requires_grad=False)
ws = w_opt.repeat([1,18,1])

print(z_samples.shape)

syn_images = G.synthesis(ws, noise_mode='const', force_fp32=True)

plot(syn_images)
