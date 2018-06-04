# Self-attention GAN implementation by Christian Cosgrove
# Based on the paper by Zhang et al.
# https://arxiv.org/abs/1805.08318

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
import model
import model_resnet
import model_mnist

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


num_classes = 10

batch_size_mult = 10

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr_gen', type=float, default=1e-4)
parser.add_argument('--lr_disc', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--load', type=str)

args = parser.parse_args()

channels = 1
width = 28

loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size*batch_size_mult, shuffle=True, num_workers=1, pin_memory=True)


Z_dim = 128
#number of updates to discriminator for every update to generator 
disc_iters = 1

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training
# if args.model == 'resnet':
#     discriminator = model_resnet.Discriminator().cuda()
#     generator = model_resnet.Generator(Z_dim).cuda()
# else:
discriminator = model_mnist.Discriminator().cuda()
generator = model_mnist.Generator(Z_dim).cuda()

if args.load is not None:
    cp_disc = torch.load(os.path.join(args.checkpoint_dir, 'disc_{}'.format(args.load)))
    cp_gen = torch.load(os.path.join(args.checkpoint_dir, 'gen_{}'.format(args.load)))
    discriminator.load_state_dict(cp_disc)
    generator.load_state_dict(cp_gen)
    print('Loaded checkpoint (epoch {})'.format(args.load))


# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_disc, betas=(0.0,0.9))
optim_gen  = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_gen, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.999)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.999)

def train(epoch):
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size*batch_size_mult:
            continue
        data, target = data.cuda(), target.cuda()

        rand_class, rand_c_onehot = make_rand_class()
        samples = data[(target == rand_class).nonzero()].squeeze()
        bsize = samples.size(0)
        data_selected = samples.repeat((args.batch_size // bsize + 1, 1,1,1,1)).view(-1, channels, width, width)[:args.batch_size]

        # update discriminator
        for _ in range(disc_iters):
            z = torch.randn(args.batch_size, Z_dim).cuda()

            optim_disc.zero_grad()
            optim_gen.zero_grad()

            disc_loss = (nn.ReLU()(1.0 - discriminator(data_selected, rand_c_onehot))).mean() + (nn.ReLU()(1.0 + discriminator(generator(z, rand_c_onehot[0]), rand_c_onehot))).mean()

            disc_loss.backward()
            optim_disc.step()

        z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
        rand_class, rand_c_onehot = make_rand_class()
        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()

        gen_loss = -discriminator(generator(z, rand_c_onehot[0]), rand_c_onehot).mean()
        gen_loss.backward()
        optim_gen.step()

        if batch_idx % 100 == 99:
            print('disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0])
    scheduler_d.step()
    scheduler_g.step()

fixed_z = torch.randn(args.batch_size, Z_dim).cuda()

def make_rand_class():
    rand_class = np.random.randint(num_classes)
    rand_c_onehot = torch.FloatTensor(args.batch_size, num_classes).cuda()
    rand_c_onehot.zero_()
    rand_c_onehot[:, rand_class] = 1
    return (rand_class, rand_c_onehot)

def evaluate(epoch):

    for fixed_class in range(num_classes):

        fixed_c_onehot = torch.FloatTensor(args.batch_size, num_classes).cuda()
        fixed_c_onehot.zero_()
        fixed_c_onehot[:, fixed_class] = 1

        samples = generator(fixed_z, fixed_c_onehot[0]).expand(-1, 3, -1, -1).cpu().detach().numpy()[:64]
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}_{}.png'.format(str(epoch).zfill(3),str(fixed_class).zfill(2)), bbox_inches='tight')
        plt.close(fig)

os.makedirs(args.checkpoint_dir, exist_ok=True)

for epoch in range(2000):
    train(epoch)
    if epoch % 10 == 0:
        evaluate(epoch)
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))

