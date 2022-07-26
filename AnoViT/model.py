import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder_r(nn.Module):
    def __init__(self, args):
        super(Decoder_r, self).__init__()
        self.args = args
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose2d(768, 256, (3, 3)),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (3, 3)),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )

        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3)),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        self.dec_block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True)
        )

        self.dec_block5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, (3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True)
        )

        self.dec_block6 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, (3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(3),
            nn.ReLU(True)
        )

        self.up = nn.UpsamplingBilinear2d((args.image_size,args.image_size)) # fixed output size
        self.tanh = nn.Tanh()
    
    def forward(self,x):
        out = x[:,1:,:]
        out = out.transpose(1,2)
        out = out.reshape(x.shape[0], -1, self.args.image_size//self.args.patch_size, self.args.image_size//self.args.patch_size)
        out = self.dec_block1(out)
        out = self.dec_block2(out)
        out = self.dec_block3(out)
        out = self.dec_block4(out)
        out = self.dec_block5(out)
        out = self.dec_block6(out)
        out = self.up(out)
        out = self.tanh(out)
        return out


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)