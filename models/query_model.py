import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import config


class LossNet(nn.Module):
    def __init__(self, feature_sizes=None, num_channels=None, interm_dim=128):
        super(LossNet, self).__init__()

        if feature_sizes is None:
            feature_sizes = [32, 16, 8, 4]
        if num_channels is None:
            num_channels = [64, 128, 256, 512]
        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out


class TDNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128, out_dim=1):
        super(TDNet, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.BN1 = nn.BatchNorm1d(interm_dim)
        self.BN2 = nn.BatchNorm1d(interm_dim)
        self.BN3 = nn.BatchNorm1d(interm_dim)
        self.BN4 = nn.BatchNorm1d(interm_dim)

        self.linear = nn.Linear(4 * interm_dim, out_dim)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        outf = torch.cat((out1, out2, out3, out4), 1)
        out = self.linear(outf)

        return out


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE_MR(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN.
    decoder 引入额外信息rank
    使用:BAAL TA-VAAL
    """

    def __init__(self, z_dim=32, nc=3, f_filt=4, num_class=10):
        super(VAE_MR, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.f_filt = f_filt
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),  # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, self.f_filt, 2, 1, bias=False),  # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024 * 2 * 2)),  # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024 * 2 * 2, z_dim)  # B, z_dim
        self.fc_logvar = nn.Linear(1024 * 2 * 2, z_dim)  # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + num_class, 1024 * 4 * 4),  # B, 1024*8*8
            View((-1, 1024, 4, 4)),  # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, self.f_filt, 2, 1, bias=False),  # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),  # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiMing_init(m)
            except:
                kaiMing_init(block)

    def forward(self, r, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z, r], 1)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
                stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class VAE_OR(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN.
    原始decoder 不引入额外信息rank
    使用:VAAL"""

    def __init__(self, z_dim=32, nc=3, f_filt=4):
        super(VAE_OR, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.f_filt = f_filt
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),  # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, self.f_filt, 2, 1, bias=False),  # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024 * 2 * 2)),  # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024 * 2 * 2, z_dim)  # B, z_dim
        self.fc_logvar = nn.Linear(1024 * 2 * 2, z_dim)  # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024 * 4 * 4),  # B, 1024*8*8
            View((-1, 1024, 4, 4)),  # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, self.f_filt, 2, 1, bias=False),  # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),  # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiMing_init(m)
            except:
                kaiMing_init(block)

    def forward(self, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Discriminator_MR(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN.
    引入额外信息rank
    使用:BAAL TA-VAAL"""

    def __init__(self, z_dim=10, num_class=10):
        super(Discriminator_MR, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_class, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiMing_init(m)

    def forward(self, r, z):
        z = torch.cat([z, r], 1)
        return self.net(z)


class Discriminator_Rfor2(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN.
    引入额外信息rank
    判别器最终输出2分类"""

    def __init__(self, z_dim=10, num_class=10):
        super(Discriminator_Rfor2, self).__init__()
        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(z_dim + num_class, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 2),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiMing_init(m)

    def forward(self, r, z):
        z = torch.cat([z, r], 1)
        return self.net(z)


class Confidnet(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN.
    引入额外信息rank
    判别器最终输出2分类"""

    def __init__(self, z_dim=10, num_class=10):
        super(Confidnet, self).__init__()
        self.z_dim = z_dim

        self.confid = nn.Sequential(
            nn.Linear(z_dim + num_class, 512),
            nn.ReLU(True),
            nn.Linear(512, 400),
            nn.ReLU(True),
            nn.Linear(400, 400),
            nn.ReLU(True),
            nn.Linear(400, 1)
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiMing_init(m)

    def forward(self, r, z):
        z = torch.cat([z, r], 1)
        uncertain = self.confid(z)
        return uncertain


def kaiMing_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Discriminator_OR(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN.
    不引入额外信息rank
    使用:VAAL"""

    def __init__(self, z_dim=10):
        super(Discriminator_OR, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiMing_init(m)

    def forward(self, z):
        return self.net(z)
