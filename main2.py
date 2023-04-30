from dataset import *
import os
from model import *

from vgg import *
from loss import *
from util import *
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import gc
import matplotlib.pyplot as plt

gc.collect()
torch.cuda.empty_cache()

MEAN=0.5
STD=0.5
batch_size=1
num_epoch = 71

weight_a = 1.5        #perpixel inner weight
weight_b =0.001     #G_SN loss weight
weight_th = 0.01       #percet loss weight
weight_v = 0.1      #tv_loss weight
weight_r = 120        #styleLoss weight
weight_e = 0.001    #d_I(gt) loss

lr =0.0001

test_dir = 'glassestest2'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([Normalization(mean=MEAN, std=STD)])

mse_loss = torch.nn.MSELoss().to(device)
l1_loss = torch.nn.L1Loss().to(device)
bce_loss = torch.nn.BCELoss().to(device)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0, 1)


# netG = SCFEGAN().to(device)
netG = SCFEGAN256().to(device)
netD = SNDiscrim().to(device)
vgg = Vgg16().to(device)

optimG = torch.optim.Adam(netG.parameters(), lr=lr)
optimD = torch.optim.Adam(netD.parameters(), lr=2 * lr)

mode = 'test'

ckpt_dir = 'ckpt_dir'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


test_gt = 'glasses_test_GT'
if not os.path.exists(test_gt):
    os.makedirs(test_gt)

test_input_dir = 'glasses_test_inputData'
if not os.path.exists(test_input_dir):
    os.makedirs(test_input_dir)

test_result_dir = 'glasses_test_result'
if not os.path.exists(test_result_dir):
    os.makedirs(test_result_dir)


if __name__=='__main__':
    if mode == 'test':
        print(device)

        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD,
                                                    optimG=optimG, optimD=optimD)

        test_dataset = ExampleDataset(test_dir, transform=transform_train)
        loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        num_data_test = len(test_dataset)

        with torch.no_grad():
            netG.eval()

            for batch, data in enumerate(loader_test):
                maked_img = data['masked_img'].to(device)
                mask = data['mask'].to(device)
                gt_img = data['label'].to(device)

                concat_img = torch.cat((maked_img, mask), dim=1)
                recon_img = netG(concat_img)

                torchvision.utils.save_image(denorm(recon_img),
                                             os.path.join(test_result_dir, 'test_result-{}.png'.format(batch)))
                torchvision.utils.save_image(denorm(maked_img),
                                             os.path.join(test_input_dir, 'test_inputData-{}.png'.format(batch)))
                torchvision.utils.save_image(denorm(gt_img),
                                             os.path.join(test_gt, 'test_GT-{}.png'.format(batch)))


                print('TEST : BATCH %04d / %04d' % (batch, num_data_test))



    image_lst = list()
    image_text = list()

    gt_image_lst = list()
    gt_image_text = list()

    fig, ax = plt.subplots(2, 10, figsize=(22,5))

    ax[0, 0].set_ylabel('Original')
    # ax[0, 0].set_ylabel('Original', rotation=0)   # y_label rotation horizontal

    ax[1, 0].set_ylabel('Image inpainting result')
    # ax[1, 0].set_ylabel('Image inpainting result', rotation=0)


    for i in range(10):
        gt_image_text.append(os.path.join(test_gt, "test_GT-{}.png".format(i)))
        gt_image_lst.append(plt.imread(gt_image_text[i]))

        ax[0, i].imshow(gt_image_lst[i])
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])

        image_text.append(os.path.join(test_result_dir, "test_result-{}.png".format(i)))
        image_lst.append(plt.imread(image_text[i]))

        ax[1, i].imshow(image_lst[i])
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])


    fig.subplots_adjust(left = 0.05, right = 0.95, wspace=0, hspace=0)
    plt.show()



