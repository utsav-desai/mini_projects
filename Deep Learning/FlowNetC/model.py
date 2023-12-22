import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from utils import conv, deconv, predict_flow, crop_like


class FlowNetC(nn.Module):
  def __init__(self, batchNorm = True):
    super(FlowNetC, self).__init__()

    self.batchNorm = batchNorm
    self.conv1 = conv(self.batchNorm, 3, 64, kernal_size = 7, stride = 2)
    self.conv2 = conv(self.batchNorm, 63, 128, kernal_size = 5, stride = 2)
    self.conv3 = conv(self.batchNorm, 128, 256, kernal_size = 5, stride = 2)
    self.conv_redir = conv(self.batchNorm, 256, 32, kernal_size = 1, stride = 1)

    self.conv3_1 = conv(self.batchNorm, 473, 256)
    self.conv4 = conv(self.batchNorm, 256, 512, stride = 2)
    self.conv4_1 = conv(self.batchNorm, 512,  512)
    self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
    self.conv5_1 = conv(self.batchNorm, 512,  512)
    self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
    self.conv6_1 = conv(self.batchNorm,1024, 1024)

    self.deconv5 = deconv(1024, 512)
    self.deconv4 = deconv(1026, 256)  # 1026 = 512 + 512 + 2
    self.deconv3 = deconv(770, 128)   # 770 = 256 + 512 + 2
    self.deconv2 = deconv(386, 64)    # 386 = 128 + 256 + 2

    self.predict_flow6 = predict_flow(1024)
    self.predict_flow5 = predict_flow(1026)
    self.predict_flow4 = predict_flow(770)
    self.predict_flow3 = predict_flow(386)
    self.predict_flow2 = predict_flow(194)

    