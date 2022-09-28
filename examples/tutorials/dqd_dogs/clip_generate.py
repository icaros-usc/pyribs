import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from einops import rearrange

import pickle
import clip
import numpy as np
from PIL import Image

from opt import AdamOpt

torch.manual_seed(22)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("USING ", device)

def norm1(prompt):
    return prompt / prompt.square().sum(dim=-1, keepdim=True).sqrt()

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def prompts_dist_loss(x, targets, loss):
    distances = [loss(x, target) for target in targets]
    return torch.stack(distances, dim=-1).sum(dim=-1)

class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, x):
        sideY, sideX = x.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = x[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)

make_cutouts = MakeCutouts(224, 32, 0.5)

def embed_image(image):
    n = image.shape[0]
    cutouts = make_cutouts(image)
    embeds = clip_model.embed_cutout(cutouts)
    embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
    return embeds

class CLIP(object):
    def __init__(self):
        clip_model_name = "ViT-B/32"
        #clip_model_name = "ViT-L/14"
        self.model, _ = clip.load(clip_model_name, device=device)
        self.model = self.model.requires_grad_(False)
        self.model.eval()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

    @torch.no_grad()
    def embed_text(self, prompt):
        return norm1(self.model.encode_text(clip.tokenize(prompt).to(device)).float())

    def embed_cutout(self, image):
        return norm1(self.model.encode_image(self.normalize(image)))

clip_model = CLIP()

model_filename = 'models/stylegan2-ffhq-1024x1024.pkl'
#model_filename = 'models/stylegan3-t-ffhqu-1024x1024.pkl'
with open(model_filename, 'rb') as fp:
    G = pickle.load(fp)['G_ema'].to(device)
    G.eval()
for p in G.parameters():
    p.requires_grad_(False)

zs = torch.randn([10000, G.mapping.z_dim], device=device)
ws = G.mapping(zs, None)
w_stds = ws.std(0)
qs = ((ws - G.mapping.w_avg) / w_stds).reshape(10000, -1)
q_norm = torch.norm(qs, dim=1).mean() * 0.15

#texts = 'A photo of Beyonce.'
#texts = 'A photo of Jennifer Lopez.'
#texts = 'A photo of a face. | A photo of Elon Musk.'
#texts = 'A photo of Elon Musk.'
texts = 'A photo of Tom Cruise.'
texts = [frase.strip() for frase in texts.split("|") if frase]
targets = [clip_model.embed_text(text) for text in texts]

def tensor_to_pil_img(img):
    img = (img.clamp(-1, 1) + 1) / 2.0
    img = img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    img = Image.fromarray(img.astype('uint8'))
    return img

with torch.inference_mode():

    batch_size = 16
    num_batches = 32
    qs = []
    losses = []
    for _ in range(num_batches):
        q = (G.mapping(torch.randn([batch_size, G.mapping.z_dim], device=device), 
                None, truncation_psi=0.7) - G.mapping.w_avg) / w_stds
        images = G.synthesis(q * w_stds + G.mapping.w_avg)
        embeds = embed_image(images.add(1).div(2))
        loss = prompts_dist_loss(embeds, targets, spherical_dist_loss).mean(0)
        i = torch.argmin(loss)
        qs.append(q[i])
        losses.append(loss[i])
    qs = torch.stack(qs)
    losses = torch.stack(losses)

    i = torch.argmin(losses)
    q = qs[i].unsqueeze(0)

output_dir = 'generations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

latent_shape = (1, -1, 512)
opt = AdamOpt(q.flatten().cpu().detach().numpy(), 0.03, betas=(0.9, 0.999))

counter = 0
while True:
    q = torch.as_tensor(opt.theta.reshape(latent_shape), device=device).requires_grad_()
    w = q * w_stds + G.mapping.w_avg
    images = G.synthesis(w, noise_mode='const')
    embeds = embed_image(images.add(1).div(2))
    
    loss = prompts_dist_loss(embeds, targets, spherical_dist_loss).mean(0)
    cur_norm = torch.norm(q)
    loss = loss + 0.01 * (q_norm - cur_norm).pow(2)
    loss.backward()

    np_grad = q.grad.cpu().detach().numpy().flatten()

    opt.step(np_grad)
    print(counter, loss[0].cpu().detach().numpy(), cur_norm)

    if counter % 5 == 0:
        img = tensor_to_pil_img(images)
        img.save(os.path.join(output_dir, f'{counter}.png'))

    counter += 1
