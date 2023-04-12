"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
import torch
import paddle
from paddle import nn
from efficientnet.model import EfficientNet
# https://github.com/GuoQuanhao/EfficientDet-Paddle/tree/main/efficientnet
from resnet import resnet18

from tools import gen_dx_bx, cumsum_trick  # , QuickCumsum


class Up(nn.Layer):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Conv2D(out_channels, out_channels, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = paddle.concat([x2, x1], axis=1)
        return self.conv(x1)


class CamEncode(nn.Layer):
    def __init__(self, D, C, downsample=0):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2D(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return paddle.nn.Softmax(axis=1)(x)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        # endpoints = dict()

        # Stem
        out, features = self.trunk._ef(x)
        # print([fet.shape for fet in features])
        x = self.up1(features[-1], features[-6])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)
        return x


class BevEncode(nn.Layer):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False)
        self.conv1 = nn.Conv2D(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias_attr=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2D(256, 128, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Layer):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        # print("init pass....")
        dx = dx.astype("float32")
        bx = bx.astype("float32")
        nx = nx.astype("float32")
        self.dx = paddle.create_parameter(shape=dx.shape,
                                          dtype=str(dx.numpy().dtype),
                                          default_initializer=paddle.nn.initializer.Assign(dx))
        self.bx = paddle.create_parameter(shape=bx.shape,
                                          dtype=str(bx.numpy().dtype),
                                          default_initializer=paddle.nn.initializer.Assign(bx))
        self.nx = paddle.create_parameter(shape=nx.shape,
                                          dtype=str(nx.numpy().dtype),
                                          default_initializer=paddle.nn.initializer.Assign(nx))

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = False

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = paddle.arange(*self.grid_conf['dbound'], dtype=paddle.float32).reshape([-1, 1, 1]).expand([-1, fH, fW])
        D, _, _ = ds.shape
        xs = paddle.linspace(0, ogfW - 1, fW, dtype=paddle.float32).reshape([1, 1, fW]).expand([D, fH, fW])
        ys = paddle.linspace(0, ogfH - 1, fH, dtype=paddle.float32).reshape([1, fH, 1]).expand([D, fH, fW])

        # D x H x W x 3
        frustum = paddle.stack((xs, ys, ds), -1)
        return paddle.create_parameter(shape=frustum.shape,
                                       dtype=str(frustum.numpy().dtype),
                                       default_initializer=paddle.nn.initializer.Assign(frustum))

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        # [41, 8, 22, 3]

        points = self.frustum - post_trans.reshape((B, N, 1, 1, 1, 3))
        # points=paddle.expand(points,[1, 1, 1,1,3, 3])
        # print(post_rots.dtype)
        # print(points.shape)
        shape_w, shape_h = points.shape[3], points.shape[4]
        points = points.reshape([points.shape[0], points.shape[1], points.shape[2], shape_w * shape_h, 3, 1])

        points = paddle.inverse(post_rots).reshape([B, N, 1, 1, 3, 3]).matmul(points)
        points = points.reshape([points.shape[0], points.shape[1], points.shape[2], shape_w, shape_h, 3])
        # print(points.shape)
        # cam_to_ego
        points = paddle.concat((points[..., :2] * points[..., 2:3],
                                points[..., 2:3]
                                ), -1)
        combine = rots.matmul(paddle.inverse(intrins))
        combine = combine.astype("float32")
        # print(combine.reshape([B, N,1,1,3, 3]).shape)
        # print(points)

        shape_w, shape_h = points.shape[3], points.shape[4]
        points = points.reshape([points.shape[0], points.shape[1], points.shape[2], shape_w * shape_h, 3, 1])

        points = combine.reshape([B, N, 1, 1, 3, 3]).matmul(points)
        points = points.reshape([points.shape[0], points.shape[1], points.shape[2], shape_w, shape_h, 3])
        points += trans.reshape([B, N, 1, 1, 1, 3])

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.reshape([B * N, C, imH, imW])
        x = self.camencode(x)
        x = x.reshape([B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample])
        x = x.transpose([0, 1, 3, 4, 5, 2])

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape([Nprime, C])

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx)
        geom_feats = geom_feats.astype("int32")
        geom_feats = geom_feats.reshape([Nprime, 3])
        batch_ix = paddle.concat([paddle.full([Nprime // B, 1], fill_value=ix, dtype="int32") for ix in range(B)])
        geom_feats = paddle.concat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = paddle.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]))
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        # collapse Z
        final = paddle.squeeze(final, axis=2)
        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):

        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)

        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)


if __name__ == "__main__":
    # net=CamEncode(41,64)
    # x=paddle.rand([24,3,128,352])
    # out=net(x)
    # print(out.shape)

    # net=BevEncode(64,1)
    # x=paddle.rand([4,64,200,200])
    # out=net(x)
    # print(out.shape)
    paddle.set_device('cpu')

    version = "mini"
    dataroot = './'
    nepochs = 10000
    gpuid = 1

    H = 900
    W = 1600
    resize_lim = (0.193, 0.225)
    final_dim = (128, 352)
    bot_pct_lim = (0.0, 0.22)
    rot_lim = (-5.4, 5.4)
    rand_flip = True
    ncams = 5
    max_grad_norm = 5.0
    pos_weight = 2.13
    logdir = './runs'

    xbound = [-50.0, 50.0, 0.5]
    ybound = [-50.0, 50.0, 0.5]
    zbound = [-10.0, 10.0, 20.0]
    dbound = [4.0, 45.0, 1.0]
    bsz = 8
    nworkers = 0
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
    # net=LiftSplatShoot(grid_conf, data_aug_conf, 1)
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    # print(model)

    st = torch.load("model525000.pt", map_location="cpu")

    from collections import OrderedDict

    new_st = {}

    for k in st.keys():
        if "num_batches_tracked" in k:
            continue
        new_st[k] = st[k]

    # model.set_dict(st)
    for ((name, param), key_t) in zip(model.named_parameters(), new_st.keys()):
        print(name, param.shape, param.dtype)
        print(key_t, new_st[key_t].shape, new_st[key_t].dtype)

        if "_fc.weight" in name:
            param.set_value(new_st[key_t].numpy().astype("float32").transpose((1, 0)))
        else:
            param.set_value(new_st[key_t].numpy().astype("float32"))
        print("**" * 10)
    print("done")
    model.eval()
    paddle.save(model.state_dict(), "torch2paddle_lss_model525000.pdparams")
