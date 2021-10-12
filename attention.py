import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
# from models import basenet

beta=1.0
class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input*beta


class Feature_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )
        # self.classifier = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) / 255
        x = self.feature(x)
        # printt('fe')
        # x = self.classifier(x)
        return x


class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(in_features=100, out_features=100)
    self.out=nn.Linear(in_features=100, out_features=10)
    # placeholder for the gradients
    self.gradients = None

  def activations_hook(self, grad):
    self.gradients = grad

  def forward(self, t):

    h = t.register_hook(self.activations_hook)
    t=RevGrad.apply(t)
    t=self.fc1(t)
    t=F.relu(t, inplace=False)

    t=self.out(t)
    return t

  def get_activations_gradient(self):
    return self.gradients

  def get_activations(self, x):
    return self.features_conv(x)


class Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(in_features=100, out_features=100)
    # self.fc2=nn.Linear(in_features=100, out_features=100)
    # self.fc3=nn.Linear(in_features=100, out_features=100)
    self.out=nn.Linear(in_features=100, out_features=10)
    # placeholder for the gradients
    self.gradients = None

  def activations_hook(self, grad):
    self.gradients = grad

  def forward(self, t):

    h = t.register_hook(self.activations_hook)
    t=self.fc1(t)
    t=F.relu(t)

    # t=self.fc2(t)
    # t=F.relu(t)

    # t=self.fc2(t)
    # t=F.relu(t)
    
    t=self.out(t)
    return t

  def get_activations_gradient(self):
    return self.gradients

  def get_activations(self, x):
    return self.features_conv(x)
    

class Dummy_Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(in_features=100, out_features=100)
    # self.fc2=nn.Linear(in_features=100, out_features=100)
    self.out=nn.Linear(in_features=100, out_features=10)
  
  def forward(self, t):

    t=self.fc1(t)
    t=F.relu(t)

    # t=self.fc2(t)
    # t=F.relu(t)

    # t=self.fc2(t)
    # t=F.relu(t)
    
    t=self.out(t)
    return t


