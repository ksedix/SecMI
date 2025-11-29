
import copy
import sys
import os
import numpy as np
import random
import tqdm
import argparse

from sklearn import metrics
from absl import flags

#this is not needed, since we are using our own model
#from model_unet import UNet
import torch
from mia_evals import resnet

# Add at the top of secmi_attack.py
import math
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
from omegaconf import OmegaConf
import sys
# Custom imports
#this is necessary so it can initialize and use the DPDM model
sys.path.insert(0, '/cephyr/NOBACKUP/groups/mlfd2025/mircog/altfl-venv')
sys.path.insert(0, '/cephyr/users/mircog/Vera/DPDM')
from model import layers, layerspp, normalization  # Replace with your module path
from model.ema import ExponentialMovingAverage
from model.ncsnpp import NCSNpp
from denoiser import EDMDenoiser, VPSDEDenoiser, VESDEDenoiser, VDenoiser
import logging
from torch.nn.parallel import DistributedDataParallel as DDP

# Add to imports at the top
import matplotlib.pyplot as plt
from datetime import datetime

from mia_evals.dataset_utils import load_member_data

from torch.utils.data import Subset

# After imports
FLAGS = None

#from utils.util import add_dimensions
class MIDataset():

    def __init__(self, member_data, nonmember_data, member_label, nonmember_label):
        self.data = torch.concat([member_data, nonmember_data])
        self.label = torch.concat([member_label, nonmember_label]).reshape(-1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, item):
        data = self.data[item]
        return data, self.label[item]


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_FLAGS(flag_path):
    global FLAGS
    FLAGS = flags.FLAGS
    flags.DEFINE_bool('train', False, help='train from scratch')
    flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
    # UNet
    flags.DEFINE_integer('ch', 128, help='base channel of UNet')
    flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
    flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
    flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
    flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
    # Gaussian Diffusion
    flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
    flags.DEFINE_float('beta_T', 0.02, help='end beta value')
    flags.DEFINE_integer('T', 1000, help='total diffusion steps')
    flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
    flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
    # Training
    flags.DEFINE_float('lr', 2e-4, help='target learning rate')
    flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
    flags.DEFINE_integer('total_steps', 800000, help='total training steps')
    flags.DEFINE_integer('img_size', 32, help='image size')
    flags.DEFINE_integer('num_channels', 3, help='number of image channels (e.g., 3 for RGB, 1 for grayscale)')
    flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
    flags.DEFINE_integer('batch_size', 128, help='batch size')
    flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
    flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
    flags.DEFINE_bool('parallel', False, help='multi gpu training')
    # Logging & Sampling
    flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
    flags.DEFINE_integer('sample_size', 64, "sampling size of images")
    flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
    # Evaluation
    flags.DEFINE_integer('save_step', 80000, help='frequency of saving checkpoints, 0 to disable during training')
    flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
    flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
    flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
    flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

    FLAGS.read_flags_from_files(flag_path)

    # Explicitly parse flags
    FLAGS([sys.argv[0]])  # This ensures flags are parsed and accessible

    return FLAGS


def calculate_auc_asr_stat(member_scores, nonmember_scores):
    print(f'member score: {member_scores.mean():.4f} nonmember score: {nonmember_scores.mean():.4f}')

    total = member_scores.size(0) + nonmember_scores.size(0)

    min_score = min(member_scores.min(), nonmember_scores.min()).item()
    max_score = min(member_scores.max(), nonmember_scores.max()).item()
    print(min_score, max_score)

    TPR_list = []
    FPR_list = []

    best_asr = 0

    TPRatFPR_1 = 0
    FPR_1_idx = 999
    TPRatFPR_01 = 0
    FPR_01_idx = 999

    for threshold in torch.range(min_score, max_score, (max_score - min_score) / 1000):
        acc = ((member_scores >= threshold).sum() + (nonmember_scores < threshold).sum()) / total

        TP = (member_scores >= threshold).sum()
        TN = (nonmember_scores < threshold).sum()
        FP = (nonmember_scores >= threshold).sum()
        FN = (member_scores < threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        ASR = (TP + TN) / (TP + TN + FP + FN)

        if ASR > best_asr:
            best_asr = ASR

        if FPR_1_idx > (0.01 - FPR).abs():
            FPR_1_idx = (0.01 - FPR).abs()
            TPRatFPR_1 = TPR

        if FPR_01_idx > (0.001 - FPR).abs():
            FPR_01_idx = (0.001 - FPR).abs()
            TPRatFPR_01 = TPR

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        print(f'Score threshold = {threshold:.16f} \t ASR: {acc:.4f} \t TPR: {TPR:.4f} \t FPR: {FPR:.4f}')
    auc = metrics.auc(np.asarray(FPR_list), np.asarray(TPR_list))
    print(f'AUC: {auc} \t ASR: {best_asr} \t TPR@FPR=1%: {TPRatFPR_1} \t TPR@FPR=0.1%: {TPRatFPR_01}')


def print_result(results):
    keys = ['auc', 'asr', 'TPR@1%FPR', 'TPR@0.1%FPR', 'threshold']
    for k, v in results.items():
        if k in keys:
            print(f'{k}: {v}')

def naive_statistic_attack(t_results, metric='l2', img_size=32, num_channels=3):
    def measure(diffusion, sample, metric, device='cuda', img_size = 32, num_channels = 3):
        diffusion = diffusion.to(device).float()
        sample = sample.to(device).float()

        if len(diffusion.shape) == 5:
            num_timestep = diffusion.size(0)
            diffusion = diffusion.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * num_channels, img_size, img_size)
            sample = sample.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * num_channels, img_size, img_size)

        if metric == 'l2':
            score = ((diffusion - sample) ** 2).flatten(1).sum(dim=-1)
        else:
            raise NotImplementedError
        return score

    # member scores
    member_scores = measure(t_results['member_diffusions'], t_results['member_internal_samples'], metric=metric, img_size=img_size, num_channels=num_channels)
    # nonmember scores
    nonmember_scores = measure(t_results['nonmember_diffusions'], t_results['nonmember_internal_samples'],
                               metric=metric, img_size=img_size, num_channels=num_channels)
    return member_scores, nonmember_scores


def execute_attack(t_result, type, img_size = 32, num_channels = 3):
    if type == 'stat':
        member_scores, nonmember_scores = naive_statistic_attack(t_result, metric='l2', img_size= img_size, num_channels= num_channels)
    elif type == 'nns':
        member_scores, nonmember_scores, model = nns_attack(t_result, train_portion=0.5, img_size= img_size, num_channels=num_channels)
        member_scores *= -1
        nonmember_scores *= -1
    else:
        raise NotImplementedError

    auc, asr, fpr_list, tpr_list, threshold = roc(member_scores, nonmember_scores, n_points=2000)
    # TPR @ 1% FPR
    tpr_1_fpr = tpr_list[(fpr_list - 0.01).abs().argmin(dim=0)]
    # TPR @ 0.1% FPR
    tpr_01_fpr = tpr_list[(fpr_list - 0.001).abs().argmin(dim=0)]

    exp_data = {
        'member_scores': member_scores,  # for histogram
        'nonmember_scores': nonmember_scores,
        'asr': asr.item(),
        'auc': auc,
        'fpr_list': fpr_list,
        'tpr_list': tpr_list,
        'TPR@1%FPR': tpr_1_fpr,
        'TPR@0.1%FPR': tpr_01_fpr,
        'threshold': threshold
    }

    return exp_data


def roc(member_scores, nonmember_scores, n_points=1000):
    max_asr = 0
    max_threshold = 0

    min_conf = min(member_scores.min(), nonmember_scores.min()).item()
    max_conf = max(member_scores.max(), nonmember_scores.max()).item()

    FPR_list = []
    TPR_list = []

    for threshold in torch.arange(min_conf, max_conf, (max_conf - min_conf) / n_points):
        TP = (member_scores <= threshold).sum()
        TN = (nonmember_scores > threshold).sum()
        FP = (nonmember_scores <= threshold).sum()
        FN = (member_scores > threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        ASR = (TP + TN) / (TP + TN + FP + FN)

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        if ASR > max_asr:
            max_asr = ASR
            max_threshold = threshold

    FPR_list = np.asarray(FPR_list)
    TPR_list = np.asarray(TPR_list)
    auc = metrics.auc(FPR_list, TPR_list)
    return auc, max_asr, torch.from_numpy(FPR_list), torch.from_numpy(TPR_list), max_threshold


def nns_attack(t_results, train_portion=0.5, device='cuda', img_size=32, num_channels=3):
    n_epoch = 15
    lr = 0.001
    batch_size = 128
    # model training
    train_loader, test_loader, num_timestep = split_nn_datasets(t_results, train_portion=train_portion,
                                                                batch_size=batch_size, img_size=img_size, num_channels=num_channels)
    print(f'num timestep: {num_timestep}')
    # initialize NNs
    model = resnet.ResNet18(num_channels=num_channels * num_timestep * 1, num_classes=1).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # model eval

    test_acc_best_ckpt = None
    test_acc_best = 0
    for epoch in range(n_epoch):
        train_loss, train_acc = nn_train(epoch, model, optim, train_loader)
        test_loss, test_acc = nn_eval(model, test_loader)
        if test_acc > test_acc_best:
            test_acc_best_ckpt = copy.deepcopy(model.state_dict())

    # resume best ckpt
    model.load_state_dict(test_acc_best_ckpt)
    model.eval()
    # generate member_scores, nonmember_scores
    member_scores = []
    nonmember_scores = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            logits = model(data.to(device))
            member_scores.append(logits[label == 1])
            nonmember_scores.append(logits[label == 0])

    member_scores = torch.concat(member_scores).reshape(-1)
    nonmember_scores = torch.concat(nonmember_scores).reshape(-1)
    return member_scores, nonmember_scores, model


def nn_train(epoch, model, optimizer, data_loader, device='cuda'):
    model.train()

    mean_loss = 0
    total = 0
    acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data = data.to(device)
        label = label.to(device).reshape(-1, 1)

        logit = model(data)

        loss = ((logit - label) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        total += data.size(0)

        logit[logit >= 0.5] = 1
        logit[logit < 0.5] = 0
        acc += (logit == label).sum()

    mean_loss /= len(data_loader)
    print(f'Epoch: {epoch} \t Loss: {mean_loss:.4f} \t Acc: {acc / total:.4f} \t')
    return mean_loss, acc / total


def split_nn_datasets(t_results, train_portion=0.5, batch_size=128, img_size=32, num_channels=3):
    # split training and testing
    # [t, 25000, 3, 32, 32]
    member_diffusion = t_results['member_diffusions']
    member_sample = t_results['member_internal_samples']
    nonmember_diffusion = t_results['nonmember_diffusions']
    nonmember_sample = t_results['nonmember_internal_samples']
    if len(member_diffusion.shape) == 4:
        # with one timestep
        # minus
        num_timestep = 1
        member_concat = (member_diffusion - member_sample).abs() ** 1
        nonmember_concat = (nonmember_diffusion - nonmember_sample).abs() ** 1

        # Oversample non-members by duplicating 5 times to match 50,000 members
        # nonmember_concat = nonmember_concat.repeat(5, 1, 1, 1)  # [10,000, 3, 32, 32] -> [50,000, 3, 32, 32]
    elif len(member_diffusion.shape) == 5:
        # with multiple timestep
        # minus
        num_timestep = member_diffusion.size(0)
        member_concat = ((member_diffusion - member_sample).abs() ** 2).permute(1, 0, 2, 3, 4).reshape(-1,
                                                                                                       num_timestep * num_channels, img_size, img_size)
        nonmember_concat = ((nonmember_diffusion - nonmember_sample).abs() ** 2).permute(1, 0, 2, 3, 4).reshape(-1,
                                                                                                                num_timestep * num_channels, img_size, img_size)
        
        # Oversample non-members
        # nonmember_concat = nonmember_concat.repeat(5, 1, 1, 1)
    else:
        raise NotImplementedError

    # train num
    num_train_members = int(member_concat.size(0) * train_portion)
    num_train_nonmembers = int(nonmember_concat.size(0) * train_portion)

    # Use the smaller number to ensure balance in training set
    num_train = min(num_train_members, num_train_nonmembers)  # e.g., min(25,000, 5,000) = 5,000

    # split
    train_member_concat = member_concat[:num_train]
    train_member_label = torch.ones(train_member_concat.size(0))
    train_nonmember_concat = nonmember_concat[:num_train]
    train_nonmember_label = torch.zeros(train_nonmember_concat.size(0))
    
    test_member_concat = member_concat[num_train:]
    test_member_label = torch.ones(test_member_concat.size(0))
    test_nonmember_concat = nonmember_concat[num_train:]
    test_nonmember_label = torch.zeros(test_nonmember_concat.size(0))

    # Debugging: Check sizes
    print(f"Train members: {train_member_concat.size(0)}, Train non-members: {train_nonmember_concat.size(0)}")
    print(f"Test members: {test_member_concat.size(0)}, Test non-members: {test_nonmember_concat.size(0)}")

    # datasets
    if num_train == 0:
        train_dataset = None
        train_loader = None
    else:
        train_dataset = MIDataset(train_member_concat, train_nonmember_concat, train_member_label,
                                  train_nonmember_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MIDataset(test_member_concat, test_nonmember_concat, test_member_label, test_nonmember_label)
    # dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_timestep


@torch.no_grad()
def nn_eval(model, data_loader, device='cuda'):
    model.eval()

    mean_loss = 0
    total = 0
    acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device).reshape(-1, 1)
        logit = model(data)

        loss = ((logit - label) ** 2).mean()

        mean_loss += loss.item()
        total += data.size(0)

        logit[logit >= 0.5] = 1
        logit[logit < 0.5] = 0

        acc += (logit == label).sum()

    mean_loss /= len(data_loader)
    print(f'Test: \t Loss: {mean_loss:.4f} \t Acc: {acc / total:.4f} \t')
    return mean_loss, acc / total


#Additions
# EDMDenoiser (unchanged from your provided code)
class EDMDenoiser(torch.nn.Module):
    def __init__(self, model, sigma_data=math.sqrt(1. / 3)):
        super().__init__()
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, x, sigma, y=None):
        # Debug: Check sigma values
        #print(f"sigma min: {sigma.min().item():.6f}, sigma max: {sigma.max().item():.6f}")
        if torch.any(sigma <= 0):
            print("Warning: sigma contains zero or negative values!")

        c_skip = self.sigma_data ** 2. / (sigma ** 2. + self.sigma_data ** 2.)
        c_out = sigma * self.sigma_data / torch.sqrt(self.sigma_data ** 2. + sigma ** 2.)
        c_in = 1. / torch.sqrt(self.sigma_data ** 2. + sigma ** 2.)
        c_noise = 0.25 * torch.log(sigma)

        # Debug: Check for nan in coefficients
        if torch.any(torch.isnan(c_noise)):
            print("c_noise contains nan!")
        if torch.any(torch.isnan(c_in)):
            print("c_in contains nan!")

        # Debug: Check input to the model
        input_to_model = c_in * x
        if torch.any(torch.isnan(input_to_model)):
            print("Input to model (c_in * x) contains nan!")

        out = self.model(input_to_model, c_noise.reshape(-1), y)

        # Debug: Check model output
        if torch.any(torch.isnan(out)):
            print("Model output contains nan!")

        x_denoised = c_skip * x + c_out * out

        # Debug: Check final output
        if torch.any(torch.isnan(x_denoised)):
            print("x_denoised contains nan!")

        return x_denoised

# Replace load_member_data
def load_dpdm_member_data(dataset_root, dataset_name, batch_size, shuffle, randaugment=False):
    transform = transforms.Compose([transforms.ToTensor()])

    # DPDM split: Full train as members, test as non-members
    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(dataset_root, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(dataset_root, train=False, transform=transform, download=True)
        num_channels = 3
    elif dataset_name == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(dataset_root, train=True, transform=transform, download=True)
        test_dataset = datasets.FashionMNIST(dataset_root, train=False, transform=transform, download=True)
        num_channels = 1
    else:
        raise NotImplementedError
    subset = Subset(train_dataset, range(10000))
    member_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    nonmember_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return None, None, member_loader, nonmember_loader


# Authors' get_model for UNet (used for secmi_unet)
def get_model(ckpt, FLAGS, WA=True, device='cuda'):
    from model_unet import UNet
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    # Load model and evaluate
    ckpt = torch.load(ckpt, map_location=device, weights_only=True)

    if WA:
        weights = ckpt['ema_model']
    else:
        weights = ckpt['net_model']

    new_state_dict = {}
    for key, val in weights.items():
        if key.startswith('module.'):
            new_state_dict.update({key[7:]: val})
        else:
            new_state_dict.update({key: val})

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# Your get_model for DPDM (used for denoiser)
def get_dpdm_model(config, device='cuda'):
    if config.model.denoiser_name == 'edm':
        if config.model.denoiser_network == 'song':
            model = EDMDenoiser(
                NCSNpp(**config.model.network).to(device))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    # Load the checkpoint
    state = torch.load(config.model.ckpt, map_location=device, weights_only=True)
    # Strip the "module." prefix from the state dict keys
    state_dict = state['model']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove the "module." prefix
        new_state_dict[new_key] = value
    # Load the modified state dict into the model
    logging.info(model.load_state_dict(new_state_dict, strict=True))

    if config.model.use_ema:
        ema = ExponentialMovingAverage(
            model.parameters(), decay=config.model.ema_rate)
        ema.load_state_dict(state['ema'])
        ema.copy_to(model.parameters())
    model.eval()
    return model


def deterministic_forward_diffusion(images, t_sec, beta_1, beta_T=0.02, T=1000, device='cuda'):
    steps = torch.arange(T, device=device, dtype=torch.float64)
    betas = torch.linspace(beta_1, beta_T, T).double().to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alpha_t = alphas_cumprod[t_sec - 1]
    noise_level = torch.sqrt(1 - alpha_t)
    noise = torch.randn_like(images, device=device)
    diffused_images = torch.sqrt(alpha_t) * images + noise_level * noise
    return diffused_images

#original secMI code

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def norm(x):
    return (x + 1) / 2

def ddim_singlestep(model, FLAGS, x, t_c, t_target, requires_grad=False, device='cuda'):

    x = x.to(device)

    t_c = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_c)
    t_target = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_target)

    betas = torch.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).double().to(device)
    alphas = 1. - betas
    alphas = torch.cumprod(alphas, dim=0)

    alphas_t_c = extract(alphas, t=t_c, x_shape=x.shape)
    alphas_t_target = extract(alphas, t=t_target, x_shape=x.shape)

    if requires_grad:
        epsilon = model(x, t_c)
    else:
        with torch.no_grad():
            epsilon = model(x, t_c)

    pred_x_0 = (x - ((1 - alphas_t_c).sqrt() * epsilon)) / (alphas_t_c.sqrt())
    x_t_target = alphas_t_target.sqrt() * pred_x_0 \
                 + (1 - alphas_t_target).sqrt() * epsilon

    return {
        'x_t_target': x_t_target,
        'epsilon': epsilon
    }


def ddim_multistep(model, FLAGS, x, t_c, target_steps, clip=False, device='cuda', requires_grad=False):
    for idx, t_target in enumerate(target_steps):
        result = ddim_singlestep(model, FLAGS, x, t_c, t_target, requires_grad=requires_grad, device=device)
        x = result['x_t_target']
        t_c = t_target
    if clip:
        result['x_t_target'] = torch.clip(result['x_t_target'], -1, 1)
    return result


# Utility function for ddim_sampler (unchanged from DPDM)
def add_dimensions(x, n_additional_dims):
    for _ in range(n_additional_dims):
        x = x.unsqueeze(-1)
    return x


def guidance_wrapper(denoiser, guid_scale):
    # Change: Removed trailing comma in parameters
    # Why: Cleaner syntax, matches Python style
    def guidance_denoiser(x, t, y):
        if guid_scale > 0:
            no_class_label = denoiser.module.model.label_dim * torch.ones_like(y, device=x.device)
            return (1. + guid_scale) * denoiser(x, t, y) - guid_scale * denoiser(x, t, no_class_label)
        else:
            return denoiser(x, t, y)
    return guidance_denoiser

def ddim_sampler(x, y, denoiser, num_steps, tmin, tmax, rho=7, guid_scale=None, stochastic=False, device='cuda'):
    # Add a small offset to tmax to avoid zero
    tmax_adjusted = tmax + 1e-6 if tmax == 0 else tmax
    # Change: Set rho=7 as default; added device='cuda'; removed **kwargs
    # Why: Make rho explicit (was implicit in DPDM), ensure cuda device for secmi_attack, remove unused kwargs for simplicity
    t_steps = torch.linspace(tmax_adjusted ** (1. / rho), tmin ** (1. / rho), steps=num_steps, device=device) ** rho
    # Change: Used device parameter instead of x.device
    # Why: Ensure all tensors on cuda, controlled by device parameter
    #print(f"t_steps: {t_steps.tolist()}")
    #x = x.to(device) * t_steps[0] #setting this to 0 will cause a bug
    x = x.to(device)
    #print(f"Initial x norm: {torch.norm(x).item():.4f}")
    # Change: Added x.to(device)
    # Why: Explicitly move x to cuda
    y = y.to(device)
    # Change: Added y.to(device)
    # Why: Explicitly move y to cuda

    if guid_scale is not None:
        denoiser = guidance_wrapper(denoiser, guid_scale)
        # Change: Kept guidance_wrapper call as in original, using the provided function
        # Why: Maintain original simplicity, avoid inline complexity

    for i, (t0, t1) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        dt = t1 - t0
        t_eval = t0 * add_dimensions(torch.ones(x.shape[0], device=device), len(x.shape) - 1)
        # Change: Used device parameter instead of x.device in add_dimensions
        # Why: Consistent device usage

        with torch.no_grad():
            # Change: Added torch.no_grad()
            # Why: Memory-efficient inference, no gradients needed for secmi_attack
            D = denoiser(x, t_eval, y)
            mse_diff = torch.mean((D - x) ** 2)
            #print(f"Step {i}, t0={t0.item():.4f}, t1={t1.item():.4f}, MSE between D and x: {mse_diff.item():.6f}")
            f = (x - D) / t0
            x = x + dt * f
            # Change: Removed stochastic branch, kept only deterministic update
            # Why: secmi_attack requires deterministic DDIM, stochastic not needed

    t_eval = t_steps[-1] * add_dimensions(torch.ones(x.shape[0], device=device), len(x.shape) - 1)
    # Change: Used device parameter instead of x.device in add_dimensions
    # Why: Consistent device usage
    with torch.no_grad():
        # Change: Added torch.no_grad()
        # Why: Memory-efficient inference
        x = denoiser(x, t_eval, y)
    return x

# Authors' get_intermediate_results (unchanged structure, adapted for both models)
def get_intermediate_results(model, FLAGS, data_loader, t_sec, timestep, is_secmi_unet=True, is_member=False, k = 1):
    import torchvision.utils  # For saving images
    import os  # For directory creation

    target_steps = list(range(0, t_sec, timestep))[1:]

    # Create directory to save images
    save_dir = "./diffusion_images"
    os.makedirs(save_dir, exist_ok=True)

    internal_diffusion_list = []
    internal_denoised_list = []
    noise_energy_list = []
    mse_diff_list = []

    max_images = 10  # Save only 10 images

    for batch_idx, (x, y) in enumerate(tqdm.tqdm(data_loader)):  # Adjusted to handle (x, y)
        x = x.cuda()

        if is_secmi_unet:
            # Save original images before scaling
            for i in range(max_images):
                torchvision.utils.save_image(
                    x[i],  # Original image, no rescaling needed
                    os.path.join("./diffusion_images", f"original_image_{i}.png")
                )
            x = x * 2 - 1  # Scale to [-1, 1] for UNet as per authors
            # UNet: Diffuse to t_sec - timestep, denoise to t_sec, then back
            x_sec = ddim_multistep(model, FLAGS, x, t_c=0, target_steps=target_steps)

            x_sec = x_sec['x_t_target']
            
            # Save diffused images at t_sec
            for i in range(max_images):
                torchvision.utils.save_image(
                    (x_sec[i] + 1) / 2,  # Rescale to [0, 1]
                    os.path.join("./diffusion_images", f"diffused_t{t_sec}_image_{i}.png")
                )
            x_sec_recon = ddim_singlestep(model, FLAGS, x_sec, t_c=target_steps[-1], t_target=target_steps[-1] + timestep)

            # Save further diffused images at t_sec + 10
            for i in range(max_images):
                torchvision.utils.save_image(
                    (x_sec_recon['x_t_target'][i] + 1) / 2,  # Rescale to [0, 1]
                    os.path.join("./diffusion_images", f"diffused_t{t_sec + 10}_image_{i}.png")
                )
            
            x_sec_recon = ddim_singlestep(model, FLAGS, x_sec_recon['x_t_target'], t_c=target_steps[-1] + timestep, t_target=target_steps[-1])

            # Save reconstructed images
            for i in range(max_images):
                torchvision.utils.save_image(
                    (x_sec_recon['x_t_target'][i] + 1) / 2,  # Rescale to [0, 1]
                    os.path.join("./diffusion_images", f"reconstructed_image_{i}.png")
                )
            break
            x_sec_recon = x_sec_recon['x_t_target']

            # Calculate noise energy: L2 norm of difference between original and diffused image
            noise = x_sec - x
            noise_energy = torch.norm(noise, p=2, dim=(1, 2, 3))
            # Calculate MSE: Mean squared error between x_sec and x_sec_recon
            mse_diff = torch.mean((x_sec - x_sec_recon) ** 2, dim=(1, 2, 3))
        else:  # DPDM model
            # DPDM: Align with UNet using ddim_sampler
            # Compute noise levels
            steps = torch.arange(1000, device='cuda', dtype=torch.float64)
            betas = FLAGS.beta_1 + (0.02 - FLAGS.beta_1) * (steps / 999)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            max_noise_level = torch.sqrt(1 - alphas_cumprod[-1])
            sigma_max = 80

            # Step 1: Diffuse to t_sec - timestep (e.g., t=90)
            t_diffuse = target_steps[-1]  # e.g., 90
            noise_level_diffuse = torch.sqrt(1 - alphas_cumprod[t_diffuse - 1])
            sigma_diffuse = (noise_level_diffuse / max_noise_level) * sigma_max * k
            print(f"Step 1: Diffusing to sigma {sigma_diffuse} (t={t_diffuse})")
            x_sec = ddim_sampler(x, y, model, num_steps=len(target_steps) + 1, tmin=sigma_diffuse, tmax=0, rho=7, device='cuda')

            # Step 2: Diffuse to t_sec (e.g., t=100)
            t_sec_noise = t_diffuse + timestep  # e.g., 100
            noise_level_sec = torch.sqrt(1 - alphas_cumprod[t_sec_noise - 1])
            sigma_sec = (noise_level_sec / max_noise_level) * sigma_max * k
            print(f"Step 2: Diffusing to sigma {sigma_sec} (t={t_sec_noise})")
            x_sec_recon = ddim_sampler(x_sec, y, model, num_steps=2, tmin=sigma_sec, tmax=sigma_diffuse, rho=7, device='cuda')

            # Step 3: Denoise back to t_sec - timestep (e.g., t=90)
            print(f"Step 3: Denoising to sigma {sigma_diffuse} (t={t_diffuse})")
            x_sec_recon = ddim_sampler(x_sec_recon, y, model, num_steps=2, tmin=sigma_diffuse, tmax=sigma_sec, rho=7, device='cuda')
            
            #print(f"Batch {batch_idx}: {'Member' if is_member else 'Non-member'} - MSE between x_sec and x_sec_recon: {mse_diff.item():.6f}")
            noise = x_sec - x
            noise_energy = torch.norm(noise, p=2, dim=(1, 2, 3))
            # Debug: Check difference between x_sec and x_sec_recon
            mse_diff = torch.mean((x_sec - x_sec_recon) ** 2, dim=(1, 2, 3))

        internal_diffusion_list.append(x_sec)
        internal_denoised_list.append(x_sec_recon)
        noise_energy_list.append(noise_energy)
        mse_diff_list.append(mse_diff)
    
    # Compute average noise energy and MSE across all images
    if not noise_energy_list:
        raise ValueError("No batches processed, noise_energy_list is empty")
    if not mse_diff_list:
        raise ValueError("No batches processed, mse_diff_list is empty")

    # Compute average noise energy across all images
    all_noise_energy = torch.cat(noise_energy_list, dim=0)
    avg_noise_energy = torch.mean(all_noise_energy).item()
    # Compute average MSE across all images
    all_mse_diff = torch.cat(mse_diff_list, dim=0)
    avg_mse_diff = torch.mean(all_mse_diff).item() 

    return {
        'internal_diffusions': torch.concat(internal_diffusion_list),
        'internal_denoise': torch.concat(internal_denoised_list),
        'avg_noise_energy': avg_noise_energy,
        'avg_mse_diff': avg_mse_diff
    }


def compare_noise_schedules(denoiser, FLAGS, member_loader, nonmember_loader, secmi_unet, t_sec=100, timestep=10, num_images=3, device='cuda', workdir=".", num_channels=3):
    steps = torch.arange(1000, device=device, dtype=torch.float64)
    betas = FLAGS.beta_1 + (0.02 - FLAGS.beta_1) * (steps / 999)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    max_noise_level = torch.sqrt(1 - alphas_cumprod[-1])
    sigma_max = 80
    noise_level = torch.sqrt(1 - alphas_cumprod[t_sec - 1])
    sigma = (noise_level / max_noise_level) * sigma_max * 0.25
    target_steps = list(range(0, t_sec, timestep))[1:]
    num_steps = len(target_steps) + 1
    member_batch = next(iter(member_loader))
    nonmember_batch = next(iter(nonmember_loader))
    member_images = member_batch[0][:num_images].to(device)
    member_labels = member_batch[1][:num_images].to(device)
    nonmember_images = nonmember_batch[0][:num_images].to(device)
    nonmember_labels = nonmember_batch[1][:num_images].to(device)
    images = torch.cat([member_images, nonmember_images], dim=0)
    labels = torch.cat([member_labels, nonmember_labels], dim=0)
    total_images = images.shape[0]
    x_sec_dpdm = ddim_sampler(images, labels, denoiser, num_steps=num_steps, tmin=sigma, tmax=0, rho=7, device=device)
    x_sec_det = deterministic_forward_diffusion(images, t_sec=t_sec, beta_1=FLAGS.beta_1, beta_T=0.02, T=1000, device=device)
    x_sec_secmi_unet = ddim_multistep(secmi_unet, FLAGS, images * 2 - 1, t_c=0, target_steps=target_steps, device=device)['x_t_target']
    noise_dpdm = x_sec_dpdm - images
    noise_det = x_sec_det - images
    noise_secmi_unet = x_sec_secmi_unet - (images * 2 - 1)
    energy_dpdm = torch.norm(noise_dpdm, p=2, dim=(1, 2, 3))
    energy_det = torch.norm(noise_det, p=2, dim=(1, 2, 3))
    energy_secmi_unet = torch.norm(noise_secmi_unet, p=2, dim=(1, 2, 3))

    print("=== DEBUG: Member and Non-member batch info ===")
    print(f"member_images.shape = {member_images.shape}")
    print(f"nonmember_images.shape = {nonmember_images.shape}")

    print(f"member_labels = {member_labels.cpu().tolist()}")
    print(f"nonmember_labels = {nonmember_labels.cpu().tolist()}")

    print(f"member_images stats: min={member_images.min().item():.4f}, max={member_images.max().item():.4f}, mean={member_images.mean().item():.4f}")
    print(f"nonmember_images stats: min={nonmember_images.min().item():.4f}, max={nonmember_images.max().item():.4f}, mean={nonmember_images.mean().item():.4f}")
    
    # Confirm indexing
    for i in range(min(5, member_images.shape[0])):
        print(f"Member image {i} label: {member_labels[i].item()}")
    for i in range(min(5, nonmember_images.shape[0])):
        print(f"Non-member image {i} label: {nonmember_labels[i].item()}")
    print("member_images.shape:", member_images.shape)
    print("nonmember_images.shape:", nonmember_images.shape)


    # Save noise energy comparison to a text file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(workdir, 'noise_comparison_output')
    os.makedirs(output_dir, exist_ok=True)
    text_output_path = os.path.join(output_dir, f'noise_energy_comparison_{timestamp}.txt')
    with open(text_output_path, 'w') as f:
        f.write("Noise Energy Comparison:\n")
        for i in range(total_images):
            label = "Member" if i < num_images else "Non-Member"
            line = (f"Image {i} ({label}): DPDM = {energy_dpdm[i].item():.4f}, "
                    f"Deterministic = {energy_det[i].item():.4f}, "
                    f"SecMI UNet = {energy_secmi_unet[i].item():.4f}\n")
            ratios = (f"  Ratios: DPDM/Det = {energy_dpdm[i].item() / energy_det[i].item():.4f}, "
                      f"DPDM/UNet = {energy_dpdm[i].item() / energy_secmi_unet[i].item():.4f}\n")
            print(line.strip())
            print(ratios.strip())
            f.write(line)
            f.write(ratios)
    print(f"Saved noise energy comparison to {text_output_path}")
    
    # Create image comparison plot
    fig, axes = plt.subplots(total_images, 4, figsize=(12, total_images * 3))
    for i in range(total_images):
        label = "Member" if i < num_images else "Non-Member"
        if num_channels == 1:
            axes[i, 0].imshow(images[i].cpu().squeeze(0).numpy(), cmap='gray')
            axes[i, 1].imshow(x_sec_dpdm[i].cpu().clamp(0, 1).squeeze(0).numpy(), cmap='gray')
            axes[i, 2].imshow(x_sec_det[i].cpu().clamp(0, 1).squeeze(0).numpy(), cmap='gray')
            axes[i, 3].imshow(((x_sec_secmi_unet[i].cpu() + 1) / 2.0).clamp(0, 1).squeeze(0).numpy(), cmap='gray')
        else:
            axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0).numpy())
            axes[i, 1].imshow(x_sec_dpdm[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy())
            axes[i, 2].imshow(x_sec_det[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy())
            axes[i, 3].imshow(((x_sec_secmi_unet[i].cpu() + 1) / 2.0).clamp(0, 1).permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f"Image {i} ({label}) Original")
        axes[i, 0].axis('off')
        axes[i, 1].set_title(f"DPDM (sigma={sigma:.2f})")
        axes[i, 1].axis('off')
        axes[i, 2].set_title(f"Deterministic (t=90)")
        axes[i, 2].axis('off')
        axes[i, 3].set_title(f"SecMI UNet (t=90)")
        axes[i, 3].axis('off')
    plt.tight_layout()
    image_output_path = os.path.join(output_dir, f'noise_comparison_{timestamp}.png')
    plt.savefig(image_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved noise comparison plot to {image_output_path}")
    plt.close(fig)
    
    # Create line plot for noise intensities
    fig, ax = plt.subplots(figsize=(10, 6))
    image_indices = np.arange(total_images)
    ax.plot(image_indices, energy_dpdm.cpu().numpy(), label='DPDM', marker='o', linestyle='-', color='blue')
    ax.plot(image_indices, energy_det.cpu().numpy(), label='Deterministic', marker='s', linestyle='-', color='orange')
    ax.plot(image_indices, energy_secmi_unet.cpu().numpy(), label='SecMI UNet', marker='d', linestyle='-', color='red')
    ax.set_xlabel('Image Index')
    ax.set_ylabel('L2 Norm of Noise')
    ax.set_title('Noise Intensity Comparison Across Schedules')
    ax.legend()
    ax.grid(True)
    lineplot_output_path = os.path.join(output_dir, f'noise_energy_lineplot_{timestamp}.png')
    plt.savefig(lineplot_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved noise energy line plot to {lineplot_output_path}")
    plt.close(fig)
    
    return {
        'images': images,
        'x_sec_dpdm': x_sec_dpdm,
        'x_sec_det': x_sec_det,
        'x_sec_secmi_unet': x_sec_secmi_unet,
        'energy_dpdm': energy_dpdm,
        'energy_det': energy_det,
        'energy_secmi_unet': energy_secmi_unet
    }


def secmi_attack(FLAGS, config, unet_model_dir, dpdm_model_dir, dataset_root, timestep=10, t_sec=100, 
                 batch_size=128, dataset='cifar10', use_unet=True, use_dpdm=True, mia_attack=False, workdir = ".", device = 'cuda', k=1, noise_comparison = False, img_size = 32, num_channels=3):
    # Load splits using authors' method
    if use_dpdm:
        _, _, dpdm_member_loader, dpdm_nonmember_loader = load_dpdm_member_data(dataset_root=dataset_root, dataset_name=dataset, 
                                                                batch_size=batch_size, shuffle=False, randaugment=False)
    
    if use_unet:
        _, _, unet_member_loader, unet_nonmember_loader = load_member_data(dataset_root=dataset_root, dataset_name=dataset, batch_size=batch_size, member_split_root="mia_evals/member_splits",
                                                                shuffle=False, randaugment=False)

    if use_unet:
        # Load SecMI UNet model
        secmi_unet = get_model(os.path.join(unet_model_dir, 'checkpoint.pt'), FLAGS, WA=True, device=device)

    if use_dpdm:
        config.model.ckpt = os.path.join(dpdm_model_dir, 'final_checkpoint.pth')
        denoiser = get_dpdm_model(config, device=device)

    # Run intermediate results based on which models are loaded
    member_results = {}
    nonmember_results = {}

    if use_unet and mia_attack:
        member_results['unet'] = get_intermediate_results(secmi_unet, FLAGS, unet_member_loader, t_sec, timestep, is_secmi_unet=True, is_member=True)
        nonmember_results['unet'] = get_intermediate_results(secmi_unet, FLAGS, unet_nonmember_loader, t_sec, timestep, is_secmi_unet=True, is_member=False)

    if use_dpdm and mia_attack:
        member_results['dpdm'] = get_intermediate_results(denoiser, FLAGS, dpdm_member_loader, t_sec, timestep, is_secmi_unet=False, is_member=True, k=k)
        nonmember_results['dpdm'] = get_intermediate_results(denoiser, FLAGS, dpdm_nonmember_loader, t_sec, timestep, is_secmi_unet=False, is_member=False, k=k)

    # Prepare results dictionaries
    t_results = {}
    
    if use_unet and mia_attack:
        t_results['unet'] = {
            'member_diffusions': member_results['unet']['internal_diffusions'],
            'member_internal_samples': member_results['unet']['internal_denoise'],
            'nonmember_diffusions': nonmember_results['unet']['internal_diffusions'],
            'nonmember_internal_samples': nonmember_results['unet']['internal_denoise'],
        }

    if use_dpdm and mia_attack:
        t_results['dpdm'] = {
            'member_diffusions': member_results['dpdm']['internal_diffusions'],
            'member_internal_samples': member_results['dpdm']['internal_denoise'],
            'nonmember_diffusions': nonmember_results['dpdm']['internal_diffusions'],
            'nonmember_internal_samples': nonmember_results['dpdm']['internal_denoise'],
        }

    # Initialize dictionary to store all results for saving
    attack_results = {
        'unet': {},
        'dpdm': {},
    }

    # Execute attacks for selected models
    if use_unet and mia_attack:
        stat_results_unet = execute_attack(t_results['unet'], type='stat', img_size= img_size, num_channels= num_channels)
        print('#' * 20 + ' SecMI_stat (UNet) ' + '#' * 20)
        print_result(stat_results_unet)
        nns_results_unet = execute_attack(t_results['unet'], type='nns', img_size= img_size, num_channels= num_channels)
        print('#' * 20 + ' SecMI_NNs (UNet) ' + '#' * 20)
        print_result(nns_results_unet)
        # Store UNet results
        attack_results['unet'] = {
            'stat_results': stat_results_unet,
            'nns_results': nns_results_unet,
            'member': {
                'avg_noise_energy': member_results['unet']['avg_noise_energy'],
                'avg_mse_diff': member_results['unet']['avg_mse_diff'],
            },
            'nonmember': {
                'avg_noise_energy': nonmember_results['unet']['avg_noise_energy'],
                'avg_mse_diff': nonmember_results['unet']['avg_mse_diff'],
            }
        }

    if use_dpdm and mia_attack:
        stat_results_dpdm = execute_attack(t_results['dpdm'], type='stat', img_size = img_size, num_channels= num_channels)
        print('#' * 20 + ' SecMI_stat (DPDM) ' + '#' * 20)
        print_result(stat_results_dpdm)
        nns_results_dpdm = execute_attack(t_results['dpdm'], type='nns', img_size= img_size, num_channels= num_channels)
        print('#' * 20 + ' SecMI_NNs (DPDM) ' + '#' * 20)
        print_result(nns_results_dpdm)
        # Store DPDM results
        attack_results['dpdm'] = {
            'stat_results': stat_results_dpdm,
            'nns_results': nns_results_dpdm,
            'member': {
                'avg_noise_energy': member_results['dpdm']['avg_noise_energy'],
                'avg_mse_diff': member_results['dpdm']['avg_mse_diff'],
            },
            'nonmember': {
                'avg_noise_energy': nonmember_results['dpdm']['avg_noise_energy'],
                'avg_mse_diff': nonmember_results['dpdm']['avg_mse_diff'],
            }
        }

    # Add noise schedule comparison if both models are used
    if use_unet and use_dpdm and noise_comparison:
        compare_noise_schedules(denoiser, FLAGS, unet_member_loader, dpdm_nonmember_loader, secmi_unet, 
                               t_sec=t_sec, timestep=timestep, num_images=3, device=device, workdir=workdir, num_channels= num_channels)
            
    # Save all results to a .pt file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(workdir, 'secmi_attack_output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'secmi_attack_results_{dataset}_{timestamp}_factor_{k}.pt')
    torch.save(attack_results, output_path)
    print(f"Saved all attack results to {output_path}")
    return attack_results


def set_logger(gfile_stream):
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet_model_dir', type=str, default=".", help='Directory containing the UNet checkpoint (checkpoint.pt)')
    parser.add_argument('--dpdm_model_dir', type=str, default=".", help='Directory containing the DPDM checkpoint (final_checkpoint.pth)')
    parser.add_argument('--config', type=str, default='config/cifar10_32/sample_eps_10.0.yaml', help='Path to the DPDM config file')
    parser.add_argument('--dataset_root', type=str, default='datasets', help='Root directory for the dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name (e.g., cifar10)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cuda)')
    parser.add_argument('--t_sec', type=int, default=100, help='Time steps for diffusion')
    parser.add_argument('--k', type=int, default=10, help='Timestep interval')
    parser.add_argument('--workdir', type=str, default="logs", help='Working directory for logs')
    parser.add_argument('--use_unet', action='store_true', help='Run attack with UNet model')
    parser.add_argument('--use_dpdm', action='store_true', help='Run attack with DPDM model')
    parser.add_argument('--mia_attack', action='store_true', help='Run MIA-attack')

    parser.add_argument('--noise_comparison', action='store_true', help='Perform noise comparison attack')
    parser.add_argument('--factor', type = float, default=1, help='Scaling constant for DPDM MIA attack')

    parser.add_argument('--img_size', type=int, default=32, help='Image size (e.g., 28 for Fashion MNIST, 32 for CIFAR-10)')
    parser.add_argument('--num_channels', type=int, default=3, help='Number of image channels (e.g., 1 for grayscale, 3 for RGB)')
    args = parser.parse_args()

    # Set up working directory and logging
    workdir = args.workdir
    make_dir(workdir)
    gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
    set_logger(gfile_stream)
    logging.info("Starting SecMI attack script")
    logging.info(args)

    fix_seed(0)
    ckpt = os.path.join(args.unet_model_dir, 'checkpoint.pt')
    flag_path = os.path.join(args.unet_model_dir, 'flagfile.txt')
    device = args.device
    FLAGS = get_FLAGS(flag_path)
    
    """
    print(FLAGS.num_channels)
    print(FLAGS.img_size)
    FLAGS.img_size = args.img_size
    FLAGS.num_channels = args.num_channels
    print(FLAGS.num_channels)
    print(FLAGS.img_size)
    """
    config = OmegaConf.load(args.config)

    # Pass both config and model directories to secmi_attack
    secmi_attack(FLAGS, config, args.unet_model_dir, args.dpdm_model_dir, dataset_root=args.dataset_root, 
                 t_sec=args.t_sec, timestep=args.k, batch_size=1024, dataset=args.dataset, 
                 use_unet=args.use_unet, use_dpdm=args.use_dpdm, mia_attack=args.mia_attack, workdir = args.workdir, device = args.device, k=args.factor, noise_comparison=args.noise_comparison,
                 img_size= args.img_size, num_channels= args.num_channels)

    gfile_stream.close()





