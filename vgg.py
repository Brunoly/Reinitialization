import math
import torch
import torch.nn as nn

def weights_init_vgg(m):
    # print('=> weights init')
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.normal_(m.weight, 0, 0.1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # nn.init.xavier_normal(m.weight)
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # Note that BN's running_var/mean are
        # already initialized to 1 and 0 respectively.
        if m.weight is not None:
            m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()



defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None, affine=True, batchnorm=True):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine
        self.feature = self.make_layers(cfg, batchnorm)
        self.dataset = dataset
        if dataset == 'cifar10' or dataset == 'cinic-10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny_imagenet':
            num_classes = 200
        else:
            raise NotImplementedError("Unsupported dataset " + dataset)
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self.apply(weights_init_vgg)
        # if pretrained:
        #     model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.dataset == 'tiny_imagenet':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg11():
    return VGG(dataset='cifar10', depth=11)
def vgg11_bn():
    return VGG(dataset='cifar10', depth=11, batchnorm=True)
def vgg13():
    return VGG(dataset='cifar10', depth=13)
def vgg13_bn():
    return VGG(dataset='cifar10', depth=13, batchnorm=True)
def vgg16():
    return VGG(dataset='cifar10', depth=16)
def vgg16_bn():
    return VGG(dataset='cifar10', depth=16, batchnorm=True)
def vgg19():
    return VGG(dataset='cifar10', depth=19)
def vgg19_bn():
    return VGG(dataset='cifar10', depth=19, batchnorm=True)