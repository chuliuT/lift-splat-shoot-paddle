"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os
import paddle
from tqdm import tqdm

from models import compile_model
from data import compile_data
import time
from tools import SimpleLoss, get_batch_iou, get_val_info

paddle.set_device("gpu")


def train(version="mini",
          dataroot='../',
          nepochs=10000,

          H=900, W=1600,
          resize_lim=(0.193, 0.225),
          final_dim=(128, 352),
          bot_pct_lim=(0.0, 0.22),
          rot_lim=(-5.4, 5.4),
          rand_flip=True,
          ncams=5,
          max_grad_norm=5.0,
          pos_weight=2.13,
          logdir='./runs',

          xbound=[-50.0, 50.0, 0.5],
          ybound=[-50.0, 50.0, 0.5],
          zbound=[-10.0, 10.0, 20.0],
          dbound=[4.0, 45.0, 1.0],

          bsz=8,
          nworkers=4,
          lr=1e-3,
          weight_decay=1e-7,
          ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    model = compile_model(grid_conf, data_aug_conf, outC=1)

    model.load_dict(paddle.load("torch2paddle_lss_model525000.pdparams"))
    # model.load_dict(paddle.load("model100.pdparams"))
    print("load done")

    model.eval()
    loss_fn = SimpleLoss(pos_weight)
    val_info = get_val_info(model, valloader, loss_fn)
    print('VAL', val_info)

    # loader = tqdm(valloader)
    # with paddle.no_grad():
    #     for batch in loader:
    #         allimgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
    #         preds = model(allimgs, rots,
    #                       trans, intrins, post_rots,
    #                       post_trans)
    #         binimgs = binimgs.astype("float32")
    #         preds = paddle.nn.Sigmoid()(preds)
    #         # loss
    #         import matplotlib.pyplot as plt
    #
    #         plt.figure()
    #         plt.subplot(121)
    #         plt.imshow(preds[0][0].numpy())
    #         plt.subplot(122)
    #         plt.imshow(binimgs[0][0].numpy())
    #         plt.show()

if __name__=="__main__":
    train()