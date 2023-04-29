from math import ceil
import torch
import torch.nn as nn
from torch.nn import functional as F
from inplace_abn import InPlaceABN

from .depthwise_seperable_conv import DepthwiseSeparableConv



class SemanticHead(nn.Module):
    """
    Semantic Head compose of three main module DPC, LSFE and MC
    Args:
    - nb_class (int) : number of classes in the dataset
    """

    def __init__(self):
        super().__init__()

        self.dpc_x32 = DPC().to(torch.device("cuda:0"))
        self.dpc_x16 = DPC().to(torch.device("cuda:0"))

        self.lsfe_x8 = LSFE().to(torch.device("cuda:0"))
        self.lsfe_x4 = LSFE().to(torch.device("cuda:0"))

        self.mc_16_to_8 = MC().to(torch.device("cuda:0"))
        self.mc_8_to_4 = MC().to(torch.device("cuda:0"))


        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets={}):
        # TODO Make a loop
        # The forward is apply in a bottom up manner
        # x32 size
        print("semantic head expected inputs: ")
        p_32 = inputs
        print("input into 32")
        print(p_32.shape)
        p_32 = self.dpc_x32(p_32)
        # [B, C, x32H, x32W] -> [B, C, x16H, x16W]
        p_32_to_merge = F.interpolate(
            p_32,
            scale_factor=(2, 2),
            mode='bilinear',
            align_corners=False)
        # [B, C, x16H, x16W] -> [B, C, x4H, x4W]
        p_32 = F.interpolate(
            p_32_to_merge,
            scale_factor=(4, 4),
            mode='bilinear',
            align_corners=False)

        # x16 size
        p_16 = inputs
        p_16 = self.dpc_x16(p_16)
        p_16_to_merge = torch.add(p_32_to_merge, p_16)
        # [B, C, x16H, x16W] -> [B, C, x4H, x4W]
        p_16 = F.interpolate(
            p_16,
            scale_factor=(4, 4),
            mode='bilinear',
            align_corners=False)
        # [B, C, x16H, x16W] -> [B, C, x8H, x8W]
        p_16_to_merge = self.mc_16_to_8(p_16_to_merge)

        # x8 size
        p_8 = inputs
        p_8 = self.lsfe_x8(p_8)
        p_8 = torch.add(p_16_to_merge, p_8)
        # [B, C, x8H, x8W] -> [B, C, x4H, x4W]
        p_8_to_merge = self.mc_8_to_4(p_8)
        # [B, C, x8H, x8W] -> [B, C, x4H, x4W]
        p_8 = F.interpolate(
            p_8,
            scale_factor=(2, 2),
            mode='bilinear',
            align_corners=False)

        # x4 size
        p_4 = inputs
        p_4 = self.lsfe_x4(p_4)
        p_4 = torch.add(p_8_to_merge, p_4)

        # Create output
        # [B, 128, x4H, x4W] -> [B, 512, x4H, x4W]
        outputs = torch.cat((p_32, p_16, p_8, p_4), dim=1)
        print("semantic head output")
        print(outputs.shape)
        return outputs


    def loss(self, inputs, targets):
        """
        Weighted pixel loss, described in the paper as :
        if loss \in worst 25% of per pixel loss then w = 4/(H*W)
        else w = 0
        We keep 25% of each image appy the weigth and then compute the mean.
        """
        # First apply cross entropy on the image.
        loss = self.cross_entropy_loss(inputs, targets)
        # sort the loss and take 25 % worst pixel
        # [B, 1, H, W] -> [B, H * W]
        loss = loss.view(loss.shape[0], -1)
        size = loss.shape[1]
        max_id = int(ceil(size * 0.25))
        sorted_loss = torch.sort(loss, descending=True).values
        kept_loss = sorted_loss[:, : max_id]
        kept_loss = kept_loss * 4 / size
        kept_loss = torch.sum(kept_loss) / loss.shape[0]
        return {
            'semantic_loss': kept_loss
        }

class LSFE(nn.Module):

    def __init__(self):
        super().__init__()
        # Separable Conv
        self.conv_1 = DepthwiseSeparableConv(768, 384, 3, padding=1)
        self.conv_2 = DepthwiseSeparableConv(384, 384, 3, padding=1)
        # Inplace BN + Leaky Relu
        self.abn_1 = InPlaceABN(384)
        self.abn_2 = InPlaceABN(384)

    def forward(self, inputs):
        # Apply first conv
        outputs = self.conv_1(inputs)
        outputs = self.abn_1(outputs)

        # Apply second conv
        outputs = self.conv_2(outputs)
        return self.abn_2(outputs)


class MC(nn.Module):

    def __init__(self):
        super().__init__()
        # Separable Conv
        self.conv_1 = DepthwiseSeparableConv(384, 384, 3, padding=1)
        self.conv_2 = DepthwiseSeparableConv(384, 384, 3, padding=1)
        # Inplace BN + Leaky Relu
        self.abn_1 = InPlaceABN(384)
        self.abn_2 = InPlaceABN(384)

    def forward(self, inputs):
        # Apply first conv
        outputs = self.conv_1(inputs)
        outputs = self.abn_1(outputs)

        # Apply second conv
        outputs = self.conv_2(outputs)
        outputs = self.abn_2(outputs)

        # Apply conv
        # outputs = self.lfse(inputs)

        # Return upsample features
        return F.interpolate(
            outputs,
            scale_factor=(2, 2),
            mode='bilinear',
            align_corners=False)

class DPC(nn.Module):

    def __init__(self):
        super().__init__()
        options = {
            'in_channels'   : 384,
            'out_channels'  : 16,
            'kernel_size'   : 1
        }
        in_place_abn_dims = 16
        self.conv_first = DepthwiseSeparableConv(dilation=(1, 6),
                                                 padding=(1, 6),
                                                 **options)
        self.iabn_first = InPlaceABN(in_place_abn_dims)
        # Branch 1
        self.conv_branch_1 = DepthwiseSeparableConv(padding=1,
                                                    **options)
        self.iabn_branch_1 = InPlaceABN(in_place_abn_dims)
        # Branch 2
        self.conv_branch_2 = DepthwiseSeparableConv(dilation=(6, 21),
                                                    padding=(6, 21),
                                                    **options)
        self.iabn_branch_2 = InPlaceABN(in_place_abn_dims)
        #Branch 3
        self.conv_branch_3 = DepthwiseSeparableConv(dilation=(18, 15),
                                                    padding=(18, 15),
                                                    **options)
        self.iabn_branch_3 = InPlaceABN(in_place_abn_dims)
        # Branch 4
        self.conv_branch_4 = DepthwiseSeparableConv(dilation=(6, 3),
                                                    padding=(6, 3),
                                                    **options)
        self.iabn_branch_4 = InPlaceABN(in_place_abn_dims)
        # Last conv
        # There is some mismatch in the paper about the dimension of this conv
        # In the paper it says "This tensor is then finally passed through a
        # 1×1 convolution with 256 output channels and forms the output of the
        # DPC module." But the overall schema shows an output of 128
        # The MC module schema also show an input of 256.
        # In order to have 512 channel at the concatenation of all layers,
        # I choosed 128 output channels
        self.conv_last = nn.Conv2d(160, 16, 1)
        self.iabn_last = InPlaceABN(16)

    def forward(self, inputs):
        # First conv
        inputs = self.conv_first(inputs)
        inputs = self.iabn_first(inputs)
        # Branch 1
        cat = torch.cat([inputs for i in range(24)], dim=0)
        cat = torch.cat([cat for i in range(24)], dim=1)
        branch_1 = self.conv_branch_1(cat)
        branch_1 = self.iabn_branch_1(branch_1)
        # Branch 2
        branch_2 = self.conv_branch_2(cat)
        branch_2 = self.iabn_branch_2(branch_2)
        # Branch 3
        branch_3 = self.conv_branch_3(cat)
        branch_3 = self.iabn_branch_3(branch_3)
        # Branch 4 (take branch 3 as input)
        cat = torch.cat([branch_3 for i in range(24)], dim=1)
        print("here")
        branch_4 = self.conv_branch_4(cat)
        print("here2")
        branch_4 = self.iabn_branch_4(branch_4)
        print("here3")
        # Concatenate
        # [B, 256, H, W] -> [B, 1280, H, W]
        cat = torch.cat([inputs for i in range(24)], dim=0)
        print(cat.shape)
        print(branch_1.shape)
        print(branch_2.shape)
        print(branch_3.shape)
        print(branch_4.shape)
        concat = torch.cat(
            (cat, branch_1, branch_2, branch_3, branch_4),
            dim=1)
        print("here4")
        # Last conv
        outputs = self.conv_last(concat)
        print("got here")
        return self.iabn_last(outputs)










