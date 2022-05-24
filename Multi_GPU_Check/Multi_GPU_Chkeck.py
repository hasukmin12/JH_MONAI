# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from monai.config import print_config
from monai.engines import create_multigpu_supervised_trainer
from monai.networks.nets import UNet
import os

# print_config()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

max_epochs = 2
lr = 1e-3
device = torch.device("cuda:0")
net = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)


def fake_loss(y_pred, y):
    return (y_pred[0] + y).sum()


def fake_data_stream():
    while True:
        yield torch.rand((10, 1, 64, 64)), torch.rand((10, 1, 64, 64))



# 1 GPU
opt = torch.optim.Adam(net.parameters(), lr)
trainer = create_multigpu_supervised_trainer(net, opt, fake_loss, [device])
trainer.run(fake_data_stream(), max_epochs=max_epochs, epoch_length=2)


# # All GPU
# opt = torch.optim.Adam(net.parameters(), lr)
# trainer = create_multigpu_supervised_trainer(net, opt, fake_loss, None)
# trainer.run(fake_data_stream(), max_epochs=max_epochs, epoch_length=2)