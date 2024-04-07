from fastai.vision.all import *
from fastai.vision.gan import *
from unet import Unet
from copy import deepcopy

@delegates(Unet)
class ConditionalUnet(Unet):
    def __init__(self, dim, num_classes=None, **kwargs):
        super().__init__(dim=dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, dim * 4)
    
    def forward(self, x, time, label=None):
        x = self.init_conv(x)
        t = self.time_mlp(time)
        if label is not None:
            t += self.label_emb(label)
            
        return super().forward_blocks(x, t)


model = ConditionalUnet(dim=32, channels=3, num_classes=10)#.cuda()
model(torch.randn(1,3,32,32),torch.tensor([1])).shape
# model(torch.randn(1,1,1,1280),torch.tensor([1])).shape