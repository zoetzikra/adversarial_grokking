# Demo codes for "Deep Networks Always Grok and Here is Why", ArXiv 2024
# Authors: Ahmed Imtiaz Humayun, Randall Balestriero, Richard Baraniuk
# Website: bit.ly/grok-adversarial
# Wandb Dashboard containing example logs: bit.ly/grok-adv-trak

import torch as ch
import torchvision
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler, AdamW
import numpy as np
import ml_collections
from tqdm import tqdm
import os
import time
import logging
import wandb
from dataloaders import cifar10_dataloaders, get_LC_samples
from models import make_resnet18k
from utils import flatten_model, add_hooks_preact_resnet18
from attacks import PGD
from local_complexity import get_intersections_for_hulls
from samplers import get_ortho_hull_around_samples, get_ortho_hull_around_samples_w_orig

import sys
import os

# Set this to the directory containing your code modules
project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)
sys.path.append(project_dir)


def add_hooks(model, config, verbose=False):
    """
    Add hooks to resnet
    """
    names, modules = flatten_model(model)
    assert len(names) == len(modules)
    norm_module = ch.nn.modules.BatchNorm2d if config.use_bn else ch.nn.modules.Identity
    layer_ids = np.asarray([i for i, each in enumerate(modules) if (type(each) == norm_module)])
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    for each in layer_ids:
        modules[each].register_forward_hook(get_activation(names[each]))
    layer_names = np.sort(np.asarray(names)[layer_ids])
    if verbose:
        print('Adding Hook to', layer_names)
    return model, layer_names, activation


def get_config():
    """hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.optimizer = 'adam'
    config.lr = 1e-3
    config.lr_schedule_flag = False
    config.train_batch_size = 256
    config.test_batch_size = 1024
    config.num_steps = 500000
    config.weight_decay = 0.
    config.label_smoothing = 0.
    config.log_steps = np.unique(
        np.logspace(0, 5.7, 20).astype(int).clip(0, 500000)
    )
    config.seed = 42
    config.use_aug = False
    config.normalize = True
    if config.normalize:
        config.dmax = 2.7537
        config.dmin = -2.4291
    else:
        config.dmax = 1
        config.dmin = 0
    config.save_model = False
    config.wandb_log = True
    config.wandb_proj = 'grok-adv'
    config.wandb_pref = 'Resnet18-CIFAR10'
    config.use_ffcv = False
    config.k = 16
    config.num_class = 10
    config.use_bn = True if config.use_ffcv else False
    config.resume_dir = None
    config.resume_step = -1
    config.compute_LC = True
    config.approx_n = 1024
    config.n_frame = 40
    config.r_frame = 0.005
    config.LC_batch_size = 256
    config.inc_centroid = False
    config.compute_robust = True
    config.atk_eps = 50/255
    config.atk_alpha = 4/255
    config.atk_itrs = 10
    return config


def train(model, loaders, config, add_hook_fn, hulls=None):
    model.cuda()
    if config.optimizer == 'sgd':
        print('Using SGD optimizer')
        opt = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError
    if config.resume_step > 0 and config.resume_dir is not None:
        assert os.path.exists(
            os.path.join(config.load_dir, f'checkpoint-s:{config.resume_step}.pt')
        ), f"Resume checkpoint not found"
        base_chkpt = ch.load(os.path.join(config.resume_dir, f'checkpoint-s:{-1}.pt'))
        model = base_chkpt['model']
        opt = base_chkpt['optimizer']
        state_chkpt = ch.load(os.path.join(config.resume_dir, f'checkpoint-s:{config.resume_step}.pt'))
        model.load_state_dict(state_chkpt['model_state_dict'])
        opt.load_state_dict(state_chkpt['optimizer_state_dict'])
    ch.save({'model': model, 'optimizer': opt}, os.path.join(config.model_dir, f'checkpoint-s:{-1}.pt'))
    iters_per_epoch = len(loaders['train'])
    epochs = np.floor(config.num_steps / iters_per_epoch)
    if config.lr_schedule_flag:
        print('Using Learning Rate Schedule')
        lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                                [0, config.lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                                [0, 1, 0])
        scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    loss_fn = CrossEntropyLoss(label_smoothing=config.label_smoothing)
    train_step = 0 if config.resume_step <= 0 else config.resume_step
    stat_names = ['train_acc', 'train_loss', 'test_loss', 'test_acc', 'adv_acc', 'train_step'] + [each+'_LC' for each in list(hulls.keys())]
    stats = dict(zip(stat_names, [[] for _ in stat_names]))
    print(f'Logging stats for steps:{config.log_steps}')
    while True:
        if train_step > config.num_steps:
            break
        for ims, labs in tqdm(loaders['train'], desc=f"train_step:{train_step}-{train_step+iters_per_epoch}"):
            ims = ims.cuda()
            labs = labs.cuda()
            opt.zero_grad()
            out = model(ims)
            loss = loss_fn(out, labs)
            loss.backward()
            opt.step()
            train_step += 1
            if config.lr_schedule_flag:
                scheduler.step()
            if train_step in config.log_steps:
                print('Computing stats...')
                model.eval()
                if config.save_model:
                    ch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict()}, os.path.join(config.model_dir, f'checkpoint-s:{train_step}.pt'))
                train_acc, train_loss = evaluate(model, loaders['train'], loss_fn)
                test_acc, test_loss = evaluate(model, loaders['test'], loss_fn)
                stats['train_acc'].append(train_acc)
                stats['test_acc'].append(test_acc)
                stats['train_loss'].append(train_loss)
                stats['test_loss'].append(test_loss)
                stats['train_step'].append(train_step)
                if config.compute_LC:
                    model, layer_names, activation_buffer = add_hook_fn(model, config)
                    if hulls is not None:
                        for k in hulls.keys():
                            with ch.no_grad():
                                n_inters, _ = get_intersections_for_hulls(
                                    hulls[k],
                                    model=model,
                                    batch_size=config.LC_batch_size,
                                    layer_names=layer_names,
                                    activation_buffer=activation_buffer
                                )
                            stats[k+'_LC'].append(n_inters.cpu())
                if config.compute_robust:
                    adv_acc = evaluate_adv(model, loaders['test'], config)
                    stats['adv_acc'].append(adv_acc)
                if config.wandb_log:
                    wandb.log({
                        'iter': train_step,
                        'train/acc': stats['train_acc'][-1],
                        'train/loss': stats['train_loss'][-1],
                        'test/acc': stats['test_acc'][-1],
                        'test/loss': stats['test_loss'][-1],
                        'train/LC': stats['train_LC'][-1].sum(1).mean(0),
                        'test/LC': stats['test_LC'][-1].sum(1).mean(0),
                        'random/LC': stats['rand_LC'][-1].sum(1).mean(0),
                        'adv/acc': stats['adv_acc'][-1]
                    })
                model.train()
    ch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict()}, os.path.join(config.model_dir, f'checkpoint-s:{train_step}.pt'))
    return stats

@ch.no_grad()
def evaluate(model, dloader, loss_fn=None):
    acc = 0
    loss = 0
    nsamples = 0
    nbatch = 0
    for ims, labs in dloader:
        ims = ims.cuda()
        labs = labs.cuda()
        outs = model(ims)
        if loss_fn is not None:
            loss += loss_fn(outs, labs)
            nbatch += 1
        acc += ch.sum(labs == outs.argmax(dim=-1)).cpu()
        nsamples += outs.shape[0]
    return acc / nsamples, loss / nbatch

def evaluate_adv(model, dloader, config):
    atk = PGD(model, eps=config.atk_eps, alpha=config.atk_alpha, steps=config.atk_itrs, dmin=config.dmin, dmax=config.dmax)
    acc = 0
    nsamples = 0
    for ims, labs in tqdm(dloader, desc=f"Computing robust acc for eps:{config.atk_eps:.3f}"):
        ims = ims.cuda()
        labs = labs.cuda()
        adv_images = atk(ims, labs)
        with ch.no_grad():
            adv_pred = model(adv_images).argmax(dim=-1)
        acc += ch.sum(labs == adv_pred).cpu()
        nsamples += adv_pred.shape[0]
    return acc / nsamples

def main():
    config = get_config()
    # Set up directories
    timestamp = time.ctime().replace(' ', '_')
    config.model_dir = os.path.join(f'./models/{timestamp}')
    config.log_dir = os.path.join(f'./logs/{timestamp}')
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    # W&B setup
    if config.wandb_log:
        wandb_project = config.wandb_proj
        wandb_run_name = f"{config.wandb_pref}-{timestamp}"
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    # Data loading
    train_loader, test_loader = cifar10_dataloaders(config)
    sampler_params = {'n': config.n_frame if not config.inc_centroid else config.n_frame+1, 'r': config.r_frame, 'seed': config.seed}
    sampler = get_ortho_hull_around_samples_w_orig if config.inc_centroid else get_ortho_hull_around_samples
    train_LC_batch, _ = get_LC_samples(train_loader, config)
    test_LC_batch, _ = get_LC_samples(train_loader, config)
    if config.normalize:
        rand_LC_batch = ch.rand_like(test_LC_batch) * 2.8 * 2 - 2.8
    else:
        rand_LC_batch = ch.rand_like(test_LC_batch)
    train_hulls = sampler(train_LC_batch.cuda(), **sampler_params).cpu()
    test_hulls = sampler(test_LC_batch.cuda(), **sampler_params).cpu()
    rand_hulls = sampler(rand_LC_batch.cuda(), **sampler_params).cpu()
    hulls = {'train': train_hulls, 'test': test_hulls, 'rand': rand_hulls}
    loaders = {'train': train_loader, 'test': test_loader}
    model = make_resnet18k(k=config.k, num_classes=config.num_class, bn=config.use_bn)
    stats = train(model, loaders, config=config, hulls=hulls, add_hook_fn=add_hooks)

if __name__ == '__main__':
    main() 