
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
import model
import os
import numpy as np

import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--load', type=str)

args = parser.parse_args()

num_classes = 10
Z_dim = 128
generator = model.Generator(Z_dim).cuda()

cp_gen = torch.load(os.path.join(args.checkpoint_dir, 'gen_{}'.format(args.load)))
generator.load_state_dict(cp_gen)
print('Loaded checkpoint (epoch {})'.format(args.load))


label1 = torch.zeros(num_classes).cuda()
label1[4] = 1
label2 = torch.zeros(num_classes).cuda()
label2[7] = 1

z = torch.randn(1, Z_dim).cuda()

for x in np.arange(0, 1, 0.01):
	image = generator(z, label1 * x + label2 * (1.0 - x)).cpu().detach().numpy()[0]
	print('image, ', image.shape)
	scipy.misc.imsave('images/test{0:.2f}.png'.format(x), image.transpose((1,2,0)))