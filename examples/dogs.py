import os
import csv
import time
from pathlib import Path

import clip

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from einops import rearrange

import pickle
import numpy as np
import matplotlib.pyplot as plt

from alive_progress import alive_bar
from PIL import Image

from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import GradientAborescenceEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap

def tensor_to_pil_img(img):
    img = (img.clamp(-1, 1) + 1) / 2.0
    img = img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    img = Image.fromarray(img.astype('uint8'))
    return img

def norm1(prompt):
    return prompt / prompt.square().sum(dim=-1, keepdim=True).sqrt()

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def cos_sim_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().mul(2)

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

class CLIP(object):
    def __init__(self, device='cpu'):
        self.device = device
        clip_model_name = "ViT-B/32"
        self.model, _ = clip.load(clip_model_name, device=device)
        self.model = self.model.requires_grad_(False)
        self.model.eval()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

    @torch.no_grad()
    def embed_text(self, prompt):
        return norm1(self.model.encode_text(clip.tokenize(prompt)
               .to(self.device)).float())

    def embed_cutout(self, image):
        return norm1(self.model.encode_image(self.normalize(image)))

    def embed_image(self, image):
        n = image.shape[0]
        cutouts = make_cutouts(image)
        embeds = self.embed_cutout(cutouts)
        embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
        return embeds

class Generator(object):

    def __init__(self, device='cpu'):
        self.device = device
        model_filename = 'models/stylegan2-ffhq-1024x1024.pkl'
        with open(model_filename, 'rb') as fp:
            print(device)
            self.model = pickle.load(fp)['G_ema'].to(device)
            self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.init_stats()
        self.latent_shape = (-1, 512)

    def init_stats(self):
        zs = torch.randn([10000, self.model.mapping.z_dim], device=self.device)
        ws = self.model.mapping(zs, None)
        self.w_stds = ws.std(0)
        qs = ((ws - self.model.mapping.w_avg) / self.w_stds).reshape(10000, -1)
        self.q_norm = torch.norm(qs, dim=1).mean() * 0.1

class Classifier(object):

    def __init__(self, gen_model, class_model, celebrity_id='Lopez'):
        self.device = gen_model.device
        self.gen_model = gen_model
        self.class_model = class_model
        self.measures = []
        if celebrity_id == 'Beyonce':
            self.init_objective('A photo of the face of Beyonce.')
            self.add_measure('A small child.', 'An elderly person.')
            self.add_measure('A person with short hair.', 
                             'A person with long hair.')
        elif celebrity_id == 'Cruise':
            self.init_objective('A photo of the face of Tom Cruise.')
            self.add_measure('A small child.', 'An elderly person.')
            self.add_measure('A person with short hair.', 
                             'A person with long hair.')
        elif celebrity_id == 'Lopez':
            self.init_objective('A photo of the face of Jennifer Lopez.')
            self.add_measure('A small child.', 'An elderly person.')
            self.add_measure('A person with short hair.', 
                             'A person with long hair.')
        elif celebrity_id == 'Musk':
            self.init_objective('A photo of the face of Elon Musk.')
            self.add_measure('A small child.', 'An elderly person.')
            self.add_measure('A person with short hair.', 
                             'A person with long hair.')
        elif celebrity_id == 'Zuckerberg':
            self.init_objective('A photo of the face of Mark Zuckerberg.')
            self.add_measure('A small child.', 'An elderly person.')
            self.add_measure('A person with short hair.', 
                             'A person with long hair.')
        else:
            print('The celebrity \"{}\" is not a valid option.'.format(celebrity_id))
            exit(0)

    def init_objective(self, text_prompt):
        texts = [frase.strip() for frase in text_prompt.split("|") if frase]
        self.obj_targets = [self.class_model.embed_text(text) for text in texts]

    def add_measure(self, positive_text, negative_text):
        texts = [frase.strip() for frase in positive_text.split("|") if frase]
        negative_targets = [self.class_model.embed_text(text) for text in texts]
        
        texts = [frase.strip() for frase in negative_text.split("|") if frase]
        positive_targets = [self.class_model.embed_text(text) for text in texts]
        
        self.measures.append((negative_targets, positive_targets))

    def find_good_start_latent(self, batch_size=16, num_batches=32):
        with torch.inference_mode():
            qs = []
            losses = []
            G = self.gen_model.model
            w_stds = self.gen_model.w_stds
            for _ in range(num_batches):
                q = (G.mapping(torch.randn([batch_size, G.mapping.z_dim], device=self.device),
                    None, truncation_psi=0.7) - G.mapping.w_avg) / w_stds
                images = G.synthesis(q * w_stds + G.mapping.w_avg)
                embeds = self.class_model.embed_image(images.add(1).div(2))
                loss = prompts_dist_loss(embeds, self.obj_targets, spherical_dist_loss).mean(0)
                i = torch.argmin(loss)
                qs.append(q[i])
                losses.append(loss[i])
            qs = torch.stack(qs)
            losses = torch.stack(losses)

            i = torch.argmin(losses)
            q = qs[i].unsqueeze(0)

        return q.flatten()

    def generate_image(self, latent_code):
        ws, _ = self.transform_to_w([latent_code])
        images = self.gen_model.model.synthesis(ws, noise_mode='const')
        return images

    def transform_to_w(self, latent_codes):
        qs = []
        ws = []
        for cur_code in latent_codes:
            q = torch.tensor(
                    cur_code.reshape(self.gen_model.latent_shape), 
                    device=self.device,
                    requires_grad=True,
                )
            qs.append(q)
            w = q * self.gen_model.w_stds + self.gen_model.model.mapping.w_avg
            ws.append(w)

        ws = torch.stack(ws, dim=0)
        return ws, qs

    def compute_objective(self, sols):
        ws, qs = self.transform_to_w(sols)

        images = self.gen_model.model.synthesis(ws, noise_mode='const')
        embeds = self.class_model.embed_image(images.add(1).div(2))
    
        loss = prompts_dist_loss(embeds, self.obj_targets, spherical_dist_loss).mean(0)
        loss = loss + 0.01 * (self.gen_model.q_norm - torch.norm(qs[0])).pow(2)
        loss.backward()

        value = loss.cpu().detach().numpy()
        jacobian = -qs[0].grad.cpu().detach().numpy()
        return value, jacobian.flatten()

    def compute_measure(self, index, sols):
        ws, qs = self.transform_to_w(sols)

        images = self.gen_model.model.synthesis(ws, noise_mode='const')
        embeds = self.class_model.embed_image(images.add(1).div(2))

        measure_targets = self.measures[index]
        pos_loss = prompts_dist_loss(embeds, measure_targets[0], cos_sim_loss).mean(0)
        neg_loss = prompts_dist_loss(embeds, measure_targets[1], cos_sim_loss).mean(0)
        loss = pos_loss - neg_loss
        loss.backward()

        value = loss.cpu().detach().numpy()
        jacobian = qs[0].grad.cpu().detach().numpy()
        return value, jacobian.flatten()

    def compute_measures(self, sols):
    
        values = []
        jacobian = []
        for i in range(len(self.measures)):
            value, jac = self.compute_measure(i, sols)
            values.append(value)
            jacobian.append(jac)

        return np.stack(values, axis=0), np.stack(jacobian, axis=0)

    def compute_all(self, sols):
        with torch.inference_mode():

            ws, qs = self.transform_to_w(sols)
            qs = torch.stack(qs, dim=0)

            images = self.gen_model.model.synthesis(ws, noise_mode='const')
            embeds = self.class_model.embed_image(images.add(1).div(2))
            
            values = []
            loss = prompts_dist_loss(embeds, self.obj_targets, spherical_dist_loss).mean(0)
            loss = loss + 0.01 * (self.gen_model.q_norm - torch.norm(qs, dim=(1,2))).pow(2)
            value = loss.cpu().detach().numpy()
            values.append(value)
            
            for i in range(len(self.measures)):
                measure_targets = self.measures[i]
                pos_loss = prompts_dist_loss(
                        embeds, 
                        measure_targets[0], 
                        cos_sim_loss,
                    ).mean(0)
                neg_loss = prompts_dist_loss(
                        embeds, 
                        measure_targets[1], 
                        cos_sim_loss
                    ).mean(0)
                loss = pos_loss - neg_loss
                value = loss.cpu().detach().numpy()
                values.append(value)

        return np.stack(values, axis=0)


def transform_obj(objs):
    # Remap the objective from minimizing [0, 20] to maximizing [0, 100]
    return (20.0-objs)*5.0

def create_optimizer(algorithm, classifier, seed):
    """Creates an optimizer based on the algorithm name.

    Args:
        algorithm (str): Name of the algorithm passed into sphere_main.
        classifier (Classifier): The models for the search.
        seed (int): Main seed or the various components.
    Returns:
        Scheduler: A ribs Scheduler for running the algorithm.
    """
    bounds = [(-0.3, 0.3), (-0.3, 0.3)]
    initial_sol = classifier.find_good_start_latent().cpu().detach().numpy()
    dim = len(initial_sol)
    batch_size = 36
    num_emitters = 1
    resolution = 200
    grid_dims = (resolution, resolution)

    # Create archive.
    if algorithm in [
            "map_elites", "map_elites_line", 
            "cma_me", "cma_mega", 
    ]:
        archive = GridArchive(grid_dims, bounds, seed=seed)
    elif algorithm in ["cma_mae", "cma_maega"]:
        archive = GridArchive(
                grid_dims, bounds, 
                archive_learning_rate=0.02,
                threshold_floor=0.0,
                seed=seed,
        )
    else:
        raise ValueError(f"Algorithm `{algorithm}` is not recognized")

    # Maintain a passive elitist archive
    passive_archive = GridArchive(grid_dims, bounds, seed=seed)
    passive_archive.initialize(dim)

    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    emitter_seeds = [None] * num_emitters if seed is None else list(
        range(seed, seed + num_emitters))
    if algorithm in ["cma_mae"]:
        emitters = [
            GradientAborescenceEmitter(archive,
                             initial_sol,
                             0.03,
                             restart_rule='basic',
                             timeout=300,
                             batch_size=batch_size,
                             seed=s) for s in emitter_seeds
        ]

    return Optimizer(archive, emitters), passive_archive

def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=100, cmap="viridis")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())

def run_experiment(algorithm,
                   trial_id,
                   classifier,
                   device,
                   init_pop=100,
                   itrs=10000,
                   outdir="logs",
                   log_freq=1,
                   log_arch_freq=1000,
                   image_monitor=False,
                   image_monitor_freq=5,
                   seed=None):
 
    # Create a directory for this specific trial.
    s_logdir = os.path.join(outdir, f"{algorithm}", f"trial_{trial_id}")
    logdir = Path(s_logdir)
    if not logdir.is_dir():
        logdir.mkdir()

    # Create a directory for logging intermediate images if the monitor is on.
    if image_monitor:
        image_monitor_freq = max(1, image_monitor_freq)
        gen_output_dir = os.path.join('generations')
        logdir = Path(gen_output_dir)
        if not logdir.is_dir():
            logdir.mkdir()
        gen_output_dir = os.path.join('generations', f"trial_{trial_id}")
        logdir = Path(gen_output_dir)
        if not logdir.is_dir():
            logdir.mkdir()

    # Create a new summary file
    summary_filename = os.path.join(s_logdir, f"summary.csv")
    if os.path.exists(summary_filename):
        os.remove(summary_filename)
    with open(summary_filename, 'w') as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(['Iteration', 'QD-Score', 'Coverage', 'Maximum', 'Average'])

    is_init_pop = algorithm in ['map_elites', 'map_elites_line']
    is_dqd = algorithm in ['cma_mega', 'cma_mega_adam', 'cma_maega']

    optimizer, passive_archive = create_optimizer(algorithm, classifier, seed)
    archive = optimizer.archive

    best = -1000
    non_logging_time = 0.0
    with alive_bar(itrs) as progress:

        if is_init_pop:
            # Sample initial population
            sols = np.array([np.random.normal(size=dim) for _ in range(init_pop)])
            sols = np.expand_dims(sols, axis=1)
            latent_codes = torch.tensor(sols, dtype=torch.float32, device=device)

            values = compute_prompts(device, latent_codes, generator, clip_model, all_prompts)

            objs = values[:,0]
            measures = values[:,1:3]

            objs = transform_obj(np.array(objs, dtype=np.float32))
            measures = np.array(measures, dtype=np.float32)

            best_gen = max(objs) 
            best = max(best, best_gen)

            # Add each solution to the archive.
            for i in range(len(sols)):
                archive.add(sols[i], objs[i], measures[i])
                passive_archive.add(sols[i], objs[i], measures[i])

        for itr in range(1, itrs + 1):
            itr_start = time.time()

            if is_dqd:
                sols = optimizer.ask(grad_estimate=True)

                objs, jacobian_obj = classifier.compute_objective(sols)
                objs = transform_obj(objs)
                best = max(best, max(objs))

                measures, jacobian_measure = classifier.compute_measures(sols)

                jacobian_obj = np.expand_dims(jacobian_obj, axis=0) 
                jacobian = np.concatenate((jacobian_obj, jacobian_measure), axis=0)
                jacobian = np.expand_dims(jacobian, axis=0)

                measures = np.transpose(measures) 
                print(measures)

                objs = objs.astype(np.float32)
                measures = measures.astype(np.float32)
                jacobian = jacobian.astype(np.float32)

                optimizer.tell(objs, measures, jacobian=jacobian)

                # Update the passive elitist archive.
                for i in range(len(sols)):
                    passive_archive.add(sols[i], objs[i], measures[i])
            
            sols = optimizer.ask()

            values = classifier.compute_all(sols)
            values = np.transpose(values)

            objs = values[:,0]
            measures = values[:,1:3]

            objs = transform_obj(np.array(objs, dtype=np.float32))
            measures = np.array(measures, dtype=np.float32)

            best_gen = max(objs) 
            best = max(best, best_gen)

            optimizer.tell(objs, measures)

            # Update the passive elitist archive.
            for i in range(len(sols)):
                passive_archive.add(sols[i], objs[i], measures[i])

            non_logging_time += time.time() - itr_start
            progress()

            print('best', best, best_gen)

            if image_monitor and itr % image_monitor_freq == 0:
                best_index = np.argmax(objs)
                latent_code = sols[best_index]

                img = classifier.generate_image(latent_code)
                img = tensor_to_pil_img(img)
                img.save(os.path.join(gen_output_dir, f'{itr}.png'))

            # Save the archive at the given frequency.
            # Always save on the final iteration.
            final_itr = itr == itrs
            if (itr > 0 and itr % log_arch_freq == 0) or final_itr:

                # Save a full archive for analysis.
                df = passive_archive.as_pandas(include_solutions = final_itr)
                df.to_pickle(os.path.join(s_logdir, f"archive_{itr:08d}.pkl"))

                # Save a heatmap image to observe how the trial is doing.
                save_heatmap(passive_archive, os.path.join(s_logdir, f"heatmap_{itr:08d}.png"))

            # Update the summary statistics for the archive
            if (itr > 0 and itr % log_freq == 0) or final_itr:
                with open(summary_filename, 'a') as summary_file:
                    writer = csv.writer(summary_file)

                    sum_obj = 0
                    num_filled = 0
                    num_bins = passive_archive.bins
                    for sol, obj, beh, idx, meta in zip(*passive_archive.data()):
                        num_filled += 1
                        sum_obj += obj
                    qd_score = sum_obj / num_bins
                    average = sum_obj / num_filled
                    coverage = 100.0 * num_filled / num_bins
                    data = [itr, qd_score, coverage, best, average]
                    writer.writerow(data)


def lsi_main(algorithm, 
             trials=5,
             init_pop=100,
             itrs=10000,
             celebrity='Lopez',
             outdir='logs',
             log_freq=1,
             log_arch_freq=1000,
             image_monitor=False,
             image_monitor_freq=5,
             seed=None):
    """Experimental tool for the StyleGAN+CLIP LSI experiments.

    Args:
        algorithm (str): Name of the algorithm.
        trials (int): Number of experimental trials to run.
        init_pop (int): Initial population size for MAP-Elites (ignored for CMA variants).
        itrs (int): Iterations to run.
        celebrity (str): Which celebrity experiment to run. Options: {Beyonce, Cruise, Lopez, Musk, Zuckerberg}. 
        outdir (str): Directory to save output.
        log_freq (int): Number of iterations between computing QD metrics and updating logs.
        log_arch_freq (int): Number of iterations between saving an archive and generating heatmaps.
        image_monitor (bool): Flags if images should be saved every few iterations.
        image_monitor_freq (int): Number of iterations between saving images.
        seed (int): Seed for the algorithm. By default, there is no seed.
    """
   
    # Create a shared logging directory for the experiments for this algorithm.
    s_logdir = os.path.join(outdir, f"{algorithm}")
    logdir = Path(s_logdir)
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()
    if not logdir.is_dir():
        logdir.mkdir()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    clip_model = CLIP(device=device)
    gen_model = Generator(device=device)
    classifier = Classifier(gen_model, clip_model, celebrity_id=celebrity)

    for cur_id in range(trials):
        run_experiment(algorithm, cur_id, classifier, device, 
                       init_pop=init_pop, itrs=itrs,
                       outdir=outdir, log_freq=log_freq, 
                       log_arch_freq=log_arch_freq, 
                       image_monitor=image_monitor, 
                       image_monitor_freq=image_monitor_freq, 
                       seed=seed)

if __name__ == "__main__":
    lsi_main("cma_mae")
