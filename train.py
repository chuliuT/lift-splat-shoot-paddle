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

          bsz=32,
          nworkers=4,
          lr=1e-1,
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
    # model.load_dict(paddle.load("torch2paddle_lss_model525000.pdparams"))
    # print("load pretrained done...")
    # model.load_dict(paddle.load("runs/model1.pdparams"))
    # print("load done")
    clip = paddle.nn.clip.ClipGradByNorm(max_grad_norm)
    opt = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr, weight_decay=weight_decay,
                                grad_clip=clip)

    loss_fn = SimpleLoss(pos_weight)

    # writer = SummaryWriter(logdir=logdir)
    val_step = 1000 if version == 'mini' else 10000

    model.train()
    counter = 0
    for epoch in range(nepochs):
        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            t0 = time.time()
            opt.clear_grad()
            preds = model(imgs,
                          rots,
                          trans,
                          intrins,
                          post_rots,
                          post_trans,
                          )
            # preds = paddle.nn.Sigmoid()(preds)
            binimgs = binimgs.astype("float32")
            loss = loss_fn(preds, binimgs)
            loss.backward()

            opt.step()
            counter += 1
            t1 = time.time()

            # if counter % 10 == 0:
            #     print(counter, loss.item())
            #     writer.add_scalar('train/loss', loss.item(), counter)

            # if counter % 50 == 0:
            #     _, _, iou = get_batch_iou(preds, binimgs)
            #     writer.add_scalar('train/iou', iou, counter)
            #     writer.add_scalar('train/epoch', epoch, counter)
            #     writer.add_scalar('train/step_time', t1 - t0, counter)
            #
            # if counter % val_step == 0:
            #     val_info = get_val_info(model, valloader, loss_fn)
            #     print('VAL', val_info)
            #     writer.add_scalar('val/loss', val_info['loss'], counter)
            #     writer.add_scalar('val/iou', val_info['iou'], counter)
            #
            # if counter % val_step == 0:
            #     model.eval()
            #     mname = os.path.join(logdir, "model{}.pdparams".format(counter))
            #     print('saving', mname)
            #     paddle.save(model.state_dict(), mname)
            #     model.train()

            ### func test
            print(counter, loss.item())

        # if counter % val_step == 0:
        model.eval()
        mname = os.path.join(logdir, "model{}.pdparams".format(epoch))
        print('saving', mname)
        paddle.save(model.state_dict(), mname)
        model.train()
            # val_info = get_val_info(model, valloader, loss_fn)
            # print(val_info)
            # _, _, iou = get_batch_iou(preds, binimgs)
            # print("iou:", iou)
            ##### test done save and load complete
            # model.eval()
            # mname = os.path.join(logdir, "model{}.pdparams".format(counter))
            # print('saving', mname)
            # paddle.save(model.state_dict(), mname)
            # model.train()
if __name__=="__main__":
    train()