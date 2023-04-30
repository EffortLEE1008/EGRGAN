import os
import numpy as np
from scipy.stats import poisson
from skimage.transform import rescale, resize

import torch
import torch.nn as nn


def save(ckpt_dir, netG, netD, optimG, optimD, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netG' : netG.state_dict(), 'netD' : netD.state_dict(),
                'optimG' : optimG.state_dict(), 'optimD' : optimD.state_dict()},
                "%s/model_epoch%d.pth" % (ckpt_dir, epoch))



def load(ckpt_dir, netG, netD, optimG, optimD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG, netD, optimG, optimD, epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    netG.load_state_dict(dict_model['netG'])
    netD.load_state_dict(dict_model['netD'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG, netD, optimG, optimD, epoch

