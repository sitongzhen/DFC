from abc import ABC
import torch
import torch.nn.functional as F
import collections
import numpy as np
from torch import nn, autograd

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        # outputs = torch.tensor(0).cuda().float()
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))##hard
            # median = np.argmax(np.array(distances))#easy
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.epsilon = 0.1
        self.gamma = 0.6
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, bn_global_x, inputs, targets):

        bn_global_x = F.normalize(bn_global_x, dim=1)
        loss = torch.tensor(0).cuda().float()
        if self.use_hard:
            for n in range(4):
                outputs = cm_hard(bn_global_x, targets, self.features, self.momentum)
                outputs /= self.temp
                loss = F.cross_entropy(outputs, targets)
        else:
            outputs = cm(bn_global_x, targets, self.features, self.momentum)
            outputs /= self.temp
            loss = F.cross_entropy(outputs, targets)
        return loss, outputs

class ImageMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ImageMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.epsilon = 0.1
        self.gamma = 0.6
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, bn_global_x, targets):

        bn_global_x = F.normalize(bn_global_x, dim=1)
        outputs = cm(bn_global_x, targets, self.features, self.momentum)
        # outputs /= self.temp
        # loss = F.cross_entropy(outputs, targets)
        return outputs, self.features