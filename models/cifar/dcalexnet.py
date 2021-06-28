# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class DcBasicBlock(nn.Module):
    def __init__(self, in_chn, out_chn, ksize, stride, padding):
        super(DcBasicBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_chn, out_chn, kernel_size=ksize, stride=stride, padding=padding, bias=True
        )
        self.bn = nn.BatchNorm2d(out_chn)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        return out

class DcAlexNet(nn.Module):
  def __init__(self, num_classes=100):
    super(DcAlexNet, self).__init__()
    self.basic1 = nn.Sequential(
        DcBasicBlock(3, 96, 3, 1, 1),
        DcBasicBlock(96, 96, 3, 1, 1),
        DcBasicBlock(96, 96, 3, 2, 1),
        DcBasicBlock(96, 192, 3, 1, 1),
        DcBasicBlock(192, 192, 3, 1, 1),
        DcBasicBlock(192, 192, 3, 2, 1)
    )
    self.basic2 = nn.Sequential(
        DcBasicBlock(192, 192, 3, 1, 0),
        DcBasicBlock(192, 192, 1, 1, 0),
        DcBasicBlock(192, num_classes, 1, 1, 0)
    )
    self.avg = nn.AvgPool2d(6, 6)

  def forward(self, x):
    out = self.basic1(x)
    out = self.basic2(out)
    out = self.avg(out)
    out = torch.flatten(out, 1)
    out = F.softmax(out, 1)
    return out


def test():
    net = DcAlexNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

#  test()

