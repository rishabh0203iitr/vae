from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import os
import pickle
from tqdm import tqdm
from datetime import datetime

import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, WeightedRandomSampler
from torch.utils.data.dataset import Subset
from torchvision import transforms as T
import torch.nn.functional as F

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from config import ex
from data.util import get_dataset, IdxDataset, ZippedDataset
from module.util import get_model
import attention
# from attention import Feature_Extractor
# from attention import Discriminator
# from attention import Classifier
# from attention import Dummy_Classifier
from util import MultiDimAverageMeter


@ex.automain
def train(
    main_tag,
    dataset_tag,
    model_tag,
    data_dir,
    log_dir,
    device,
    target_attr_idx,
    bias_attr_idx,
    main_num_steps,
    main_valid_freq,
    main_batch_size,
    main_optimizer_tag,
    main_learning_rate,
    main_weight_decay,
):
    torch.manual_seed(1)
    np.random.seed(1)

    print(dataset_tag)


    start_time = datetime.now()

    device = torch.device(2)
    log_dir = "/home/vinodkk/codes/fair_lr/GAN_vis_test_dis/vis_cmnist/debias/log"
    log_writer = SummaryWriter('/home/vinodkk/codes/fair_lr/GAN_vis_test_dis/vis_cmnist/debias/log/scalar')
    # alpha_D=1e
    # alpha_C=1
    # alpha=1e2

    writer = SummaryWriter(os.path.join(log_dir, "summary", main_tag))


    # data_dir='home/vinodkk/codes/fair_lr/colour_MNIST/datasets/debias'
    train_dataset = get_dataset(
        dataset_tag,
        data_dir=data_dir,
        dataset_split="train",
        transform_split="train"
    )

    valid_dataset = get_dataset(
        dataset_tag,
        data_dir=data_dir,
        dataset_split="eval",
        transform_split="eval"
    )

    train_target_attr = train_dataset.attr[:, target_attr_idx]
    train_bias_attr = train_dataset.attr[:, bias_attr_idx]
    attr_dims = []
    attr_dims.append(torch.max(train_target_attr).item() + 1)
    attr_dims.append(torch.max(train_bias_attr).item() + 1)
    num_classes = attr_dims[0]

    train_dataset = IdxDataset(train_dataset)
    valid_dataset = IdxDataset(valid_dataset)

    best=0.0
    best2=0.0

    train_loader = DataLoader(
        train_dataset,
        batch_size=main_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )



    manualSeed = 1
    #manualSeed = random.randint(1, 10000) # use if you want new results
    # print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    vae = get_model(model_tag, 2).to(device)

    lr = 1e-3

    optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

    ### Training function
    def train_epoch(vae, device, dataloader, optimizer):
        # Set train mode for both the encoder and the decoder
        vae.train()
        train_loss = 0.0
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for x, _ in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            x_hat = vae(x)
            # Evaluate loss
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print batch loss
            print('\t partial train loss (single batch): %f' % (loss.item()))
            train_loss+=loss.item()

        return train_loss / len(dataloader.dataset)


    ## # Testing function
    def test_epoch(vae, device, dataloader):
        # Set evaluation mode for encoder and decoder
        vae.eval()
        val_loss = 0.0
        with torch.no_grad(): # No need to track the gradients
            for x, _ in dataloader:
                # Move tensor to the proper device
                x = x.to(device)
                # Encode data
                encoded_data = vae.encoder(x)
                # Decode data
                x_hat = vae(x)
                loss = ((x - x_hat)**2).sum() + vae.encoder.kl
                val_loss += loss.item()

        return val_loss / len(dataloader.dataset)


    def plot_ae_outputs(encoder,decoder,n=5):
        plt.figure(figsize=(10,4.5))
        for i in range(n):
          ax = plt.subplot(2,n,i+1)
          img = test_dataset[i][0].unsqueeze(0).to(device)
          encoder.eval()
          decoder.eval()
          with torch.no_grad():
             rec_img  = decoder(encoder(img))
          plt.imshow(img.cpu().squeeze().numpy())
          ax.get_xaxis().set_visible(False)
          ax.get_yaxis().set_visible(False)
          if i == n//2:
            ax.set_title('Original images')
          ax = plt.subplot(2, n, i + 1 + n)
          plt.imshow(rec_img.cpu().squeeze().numpy())
          ax.get_xaxis().set_visible(False)
          ax.get_yaxis().set_visible(False)
          if i == n//2:
             ax.set_title('Reconstructed images')

        plt.savefig("test1.png")


    num_epochs = 1

    for epoch in range(num_epochs):
       train_loss = train_epoch(vae,device,train_loader,optim)
       val_loss = test_epoch(vae,device,valid_loader)
       print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
       plot_ae_outputs(vae.encoder,vae.decoder,n=5)







