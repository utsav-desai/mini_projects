import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from spatial_correlation_sampler import spatial_correlation_sample



def conv(batchNorm, in_planes, out_planes, kernal_size = 3, stride = 1):
  if batchNorm:
    return nn.Sequential(
        nn.Conv2D(in_planes, out_planes, kernal_size = kernal_size, stride = stride, bias = False, padding = (kernal_size - 1) // 2),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1, inplace = True)
    )
  else:
    return nn.Sequential(
        nn.Conv2D(in_planes, out_planes, kernal_size = kernal_size, stride = stride, bias = False, padding = (kernal_size - 1) // 2),
        nn.LeakyReLU(0.1, inplace = True)
    )
  

def deconv(in_planes, out_planes):
  return nn.Sequential(
      nn.ConvTranspose2D(in_planes, out_planes, kernal_size = 4, stride = 2, padding = 1, bias = False),
      nn.LeakyReLU(0.1, inplace = True)
  )


def predict_flow(in_planes):
  return nn.Sequential(
    nn.Conv2d(in_planes, 2, kernal_size = 3, stride = 1, padding = 1, bias = False)
  )


def crop_like(input, target):
  if input.size()[2:] == target.size()[2:]:
    return input
  else:
    return input[:, :, target.size(2), target.size(3)]
  

def correlate(input1, input2):
  out_corr = spatial_correlation_sample(
    input1,
    input2,
    kernal_size = 1,
    patch_size = 21,
    stride = 1,
    padding = 0,
    dilation_patch = 2
  )
  b, ph, pw, h, w = out_corr.size()
  out_corr = out_corr.view(b, ph * pw, h, w) / input1.size(1)
  return torch.nn.functional.leaky_relu_(out_corr, 0.1)