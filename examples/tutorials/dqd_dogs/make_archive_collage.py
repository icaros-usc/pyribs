# A small script for generating a collage of faces from the QD-Archive

import matplotlib
import matplotlib.font_manager
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["font.family"] = 'serif'
matplotlib.rcParams["font.serif"] = 'Palatino'
matplotlib.rc('font', size=20)

import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import pickle

from PIL import Image

# Note that only final archives encode latent codes.
archive_filename = 'logs/cma_maega/trial_0/archive_00010000.pkl'
#archive_filename = 'results/cruise_logs/cma_maega/trial_1/archive_00010000.pkl'

# min and max index for rows then columns (row major).
# The archive is shape (200, 200) indexed from [0, 200).
archive_dims = (200, 200)
#archive_index_range = ((70, 130), (70, 130))
archive_index_range = ((90, 130), (50, 110))
# Measure ranges
measure_ranges = ((-0.3, 0.3), (-0.3, 0.3))
# Controls that x rows and y columns are generated
# Images are "evenly" (as possible) sampled based on this criteria
picture_frequency = (8, 5) 

# Use the CPU while we are running exps.
#device = "cpu"
device = "cuda"

# Uncomment to save all grid images separately.
gen_output_dir = os.path.join('grid_imgs')
logdir = Path(gen_output_dir)
if not logdir.is_dir():
    logdir.mkdir()


model_filename = 'models/stylegan2-ffhq-1024x1024.pkl'
with open(model_filename, 'rb') as fp:
    model = pickle.load(fp)['G_ema'].to(device)
    model.eval()
latent_shape = (1, -1, 512)

zs = torch.randn([10000, model.mapping.z_dim], device=device)
ws = model.mapping(zs, None)
w_stds = ws.std(0)
qs = ((ws - model.mapping.w_avg) / w_stds).reshape(10000, -1)
q_norm = torch.norm(qs, dim=1).mean() * 0.1

# Read the archive from the log (pickle file)
df = pd.read_pickle(archive_filename)

imgs = []
for j in reversed(range(picture_frequency[1])):
    for i in range(picture_frequency[0]):
        
        delta_i = archive_index_range[0][1] - archive_index_range[0][0]
        delta_j = archive_index_range[1][1] - archive_index_range[1][0]
        index_i_lower = int(delta_i * i / picture_frequency[0] + archive_index_range[0][0])
        index_i_upper = int(delta_i * (i+1) / picture_frequency[0] + archive_index_range[0][0])
        index_j_lower = int(delta_j * j / picture_frequency[1] + archive_index_range[1][0])
        index_j_upper = int(delta_j * (j+1) / picture_frequency[1] + archive_index_range[1][0])
        print(i, j, index_i_lower, index_i_upper, index_j_lower, index_j_upper)

        query_string = f"{index_i_lower} <= index_0 & index_0 <= {index_i_upper} &"
        query_string += f"{index_j_lower} <= index_1 & index_1 <= {index_j_upper}" 
        print(query_string)
        df_cell = df.query(query_string)
        
        if not df_cell.empty:

            sol = df_cell.iloc[df_cell['objective'].argmax()]
            print(sol)

            q = torch.tensor(
                    sol[5:].values.reshape(latent_shape),
                    device=device,
                    requires_grad=True,
                )
            w = q * w_stds + model.mapping.w_avg

            img = model.synthesis(w, noise_mode='const')
            img = (img.clamp(-1, 1) + 1) / 2.0 # Normalize from [0,1]

            # Uncomment to save all grid images separately.
            pil_img = img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
            pil_img = Image.fromarray(pil_img.astype('uint8'))
            pil_img.save(os.path.join(gen_output_dir, f'{j}_{i}.png'))

            img = img[0].detach().cpu()
            imgs.append(img)
        else:
            imgs.append(torch.zeros((3,1024,1024)))

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

plt.figure(figsize=(16,10))
img_grid = make_grid(imgs, nrow=picture_frequency[0], padding=0)
img_grid = np.transpose(img_grid.cpu().numpy(), (1,2,0))
plt.imshow(img_grid)

plt.xlabel("Age.")
plt.ylabel("Hair Length.")

def create_archive_tick_labels(axis_range, measure_range, dim, num_ticks):
    low_pos = axis_range[0] / dim
    high_pos = axis_range[1] / dim

    tick_offset = [
        (high_pos - low_pos) * (p / num_ticks) + low_pos
        for p in range(num_ticks+1) 
    ]
    ticklabels = [
        round((measure_range[1]-measure_range[0]) * p + measure_range[0], 2)
        for p in tick_offset
    ]
    return ticklabels

num_x_ticks = 6
num_y_ticks = 6
x_ticklabels = create_archive_tick_labels(archive_index_range[0], 
                    measure_ranges[0], archive_dims[0], num_x_ticks)
y_ticklabels = create_archive_tick_labels(archive_index_range[1],
                    measure_ranges[1], archive_dims[1], num_y_ticks)
y_ticklabels.reverse()

x_tick_range = img_grid.shape[1]
x_ticks = np.arange(0, x_tick_range+1e-9, step=x_tick_range/num_x_ticks)
y_tick_range = img_grid.shape[0]
y_ticks = np.arange(0, y_tick_range+1e-9, step=y_tick_range/num_y_ticks)
plt.xticks(x_ticks, x_ticklabels)
plt.yticks(y_ticks, y_ticklabels)
plt.tight_layout()
plt.savefig('collage.png')
