from torchvision.models.resnet import BasicBlock, Bottleneck
from opacus.validators import ModuleValidator
import torch

def make_resnet_dp_compatible(model):
    # 1. 替换 BatchNorm 为 GroupNorm
    model = ModuleValidator.fix(model)

    # 2. 禁用所有 ReLU 的 inplace（包括模型顶层的 self.relu）
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

    # 3. 猴子补丁修复残差连接中的 out += identity
    def patch_block_inplace(block_class):
        if not hasattr(block_class, "_dp_patched"):
            original_forward = block_class.forward

            def new_forward(self, x):
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                if hasattr(self, 'conv3'):
                    out = self.conv3(out)
                    out = self.bn3(out)
                else:
                    out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out = out + identity  # 关键：改为非原地加法
                out = self.relu(out)

                return out

            block_class.forward = new_forward
            block_class._dp_patched = True  # 标记已 patch，避免重复

    # patch BasicBlock 和 Bottleneck
    patch_block_inplace(BasicBlock)
    patch_block_inplace(Bottleneck)

    return model