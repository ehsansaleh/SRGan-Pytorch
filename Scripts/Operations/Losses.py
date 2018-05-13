import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

if torch.__version__.startswith('0.3'):
    from torch.nn.modules.module import Module
    class BCEWithLogitsLoss(Module):
        r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
        class. This version is more numerically stable than using a plain `Sigmoid`
        followed by a `BCELoss` as, by combining the operations into one layer,
        we take advantage of the log-sum-exp trick for numerical stability.

        This Binary Cross Entropy between the target and the output logits
        (no sigmoid applied) is:

        .. math:: loss(o, t) = - 1/n \sum_i (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))

        or in the case of the weight argument being specified:

        .. math:: loss(o, t) = - 1/n \sum_i weight[i] * (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))

        This is used for measuring the error of a reconstruction in for example
        an auto-encoder. Note that the targets `t[i]` should be numbers
        between 0 and 1.

        Args:
            weight (Tensor, optional): a manual rescaling weight given to the loss
                of each batch element. If given, has to be a Tensor of size
                "nbatch".
            size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                size_average is set to ``False``, the losses are instead summed for
                each minibatch. Default: ``True``

         Shape:
             - Input: :math:`(N, *)` where `*` means, any number of additional
               dimensions
             - Target: :math:`(N, *)`, same shape as the input

         Examples::

             >>> loss = nn.BCEWithLogitsLoss()
             >>> input = autograd.Variable(torch.randn(3), requires_grad=True)
             >>> target = autograd.Variable(torch.FloatTensor(3).random_(2))
             >>> output = loss(input, target)
             >>> output.backward()
        """
        def __init__(self, weight=None, size_average=True, reduce=True):
            super(BCEWithLogitsLoss, self).__init__()
            self.size_average = size_average
            self.reduce=reduce
            self.register_buffer('weight', weight)

        def forward(self, input, target):
            if self.weight is not None:
                if self.reduce:
                    return F.binary_cross_entropy_with_logits(input, target, Variable(self.weight), self.size_average).sum()
                else:
                    return F.binary_cross_entropy_with_logits(input, target, Variable(self.weight), self.size_average)
            else:
                if self.reduce:
                    return F.binary_cross_entropy_with_logits(input, target, size_average=self.size_average).sum()
                else:
                    return F.binary_cross_entropy_with_logits(input, target, size_average=self.size_average)
else:
    BCEWithLogitsLoss = nn.BCEWithLogitsLoss


if torch.__version__.startswith('0.3'):
    from torch.nn.modules.module import Module
    def _assert_no_grad(variable):
        assert not variable.requires_grad,             "nn criterions don't compute the gradient w.r.t. targets - please "             "mark these variables as volatile or not requiring gradients"

    class _Loss(Module):
        def __init__(self, size_average=True, reduce=True):
            super(_Loss, self).__init__()
            self.size_average = size_average
            self.reduce=reduce

    class SoftMarginLoss(_Loss):
        r"""Creates a criterion that optimizes a two-class classification
        logistic loss between input `x` (a 2D mini-batch Tensor) and
        target `y` (which is a tensor containing either `1` or `-1`).

        ::

            loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x.nelement()

        The normalization by the number of elements in the input can be disabled by
        setting `self.size_average` to ``False``.
        """
        def forward(self, input, target, reduce=True):
            _assert_no_grad(target)
            if self.reduce:
                return F.soft_margin_loss(input, target, size_average=self.size_average).sum()
            else:
                return F.soft_margin_loss(input, target, size_average=self.size_average)
else:
    SoftMarginLoss = nn.SoftMarginLoss


    
class VGGFeatureExtractor(nn.Module):
    def __init__(self, vggname='vgg19', i=5, j=4):
        super(VGGFeatureExtractor, self).__init__()
        
        creators={'vgg11':torchvision.models.vgg11,
         'vgg11_bn':torchvision.models.vgg11_bn,
         'vgg13':torchvision.models.vgg13,
         'vgg13_bn':torchvision.models.vgg13_bn,
         'vgg16':torchvision.models.vgg16,
         'vgg16_bn':torchvision.models.vgg16_bn,
         'vgg19_bn':torchvision.models.vgg19_bn,
         'vgg19':torchvision.models.vgg19}

        vggnet = creators[vggname](pretrained=True)

        if '11' in vggname:
            config_key='A'
        elif '13' in vggname:
            config_key='B'
        elif '16' in vggname:
            config_key='D'
        elif '19' in vggname:
            config_key='E'
        else:
            raise 'Unknown vgg net name'

        layers_cfg = torchvision.models.vgg.cfg[config_key]
        pool_idxs = [i for i, name in enumerate(layers_cfg) if name == 'M']

        conv_pre_layers = pool_idxs[i-2]+1

        if 'bn' in vggname:
            layeridx = conv_pre_layers * 3 + j
        else:
            layeridx = conv_pre_layers * 2 + j
        
        self.features = nn.Sequential(*list(vggnet.features.children())[:layeridx])

    def forward(self, x):
        return self.features(x)