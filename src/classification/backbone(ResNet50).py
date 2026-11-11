from__future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNEt50_weights,resnet50

class DineTuneREsNet50(nn.Module):

output_dim = 2048
def _init_(self,pretrained:bool = Ture) ->None:
  super()._init_()
  weights = ResNet50(Weights=weights)
  self.model=resnet50(weights=weights)
  self.model.fc = nn.Identify()

  for parameter in self.model.parameters():
      parameter.requires_grad_(False)
  for parameter in self.model.layer4[2].parameters():
      parameter.requires_grad_(Ture)
  self._freeze_batch_norm()
def_freeze_batch_norm(self) ->None
  for module in self.model.modules():
    if ininstance(module, nn.modules.batchnorm._BatchNorm):
      module.eval()
      for parameter in module.parameters():
          parameter.requires_gard(False)

def train(self, mode:bool = Ture) -> "FineTuneResNet50":
    super(),train(mode)
    self._freeze_batch_norm()
    return self

def forward(self, images: torch.Tensor) -> torch.Tensor:
  return self.model(images)

def trainable_parameter_name(self) -> tuple[str，...]：
  return tuple(name for name, value in self.named_parameters() if value.requires_grad)
