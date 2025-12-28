import torch
import torch.nn as nn
import torch.nn.functional as F
from torchjpeg import dct


def bdct(x, sub_channels=None, size=8, stride=8, pad=0, dilation=1):
    x = x * 0.5 + 0.5  # x to [0, 1]

    # upsample to 896x896
    x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
    x *= 255 # x to [0, 255]
    if x.shape[1] == 3:
        x = dct.to_ycbcr(x)
    x -= 128  # x to [-128, 127] 中心化处理
    bs, ch, h, w = x.shape 
    block_num = h // stride
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad,
                 stride=(stride, stride)) # 把图像切成块 ( bs*ch, size*size, block_num*block_num )
    x = x.transpose(1, 2) # 交换维度 ( bs*ch, block_num*block_num, size*size )
    x = x.view(bs, ch, -1, size, size) # [批次, 通道, 块数量, 块高度, 块宽度]
    dct_block = dct.block_dct(x)
    # 重塑重排为 (bs, ch, size*size, block_num, block_num) size*size是每个块的DCT系数数量
    dct_block = dct_block.view(bs, ch, block_num, block_num, size * size).permute(0, 1, 4, 2, 3)
    # 最终重塑为 (bs, ch*size*size, block_num, block_num)，类似特征图格式，便于CNN处理
    dct_block = dct_block.reshape(bs, -1, block_num, block_num)

    return dct_block

# 将DCT系数块转换回图像
def ibdct(x, size=8, stride=8, pad=0, dilation=1):
    bs, _, _, _ = x.shape
    sampling_rate = 8

    x = x.view(bs, 3, 64, 14 * sampling_rate, 14 * sampling_rate) # 还原通道和块维度，3是YCbCr三个通道，64是每个块的DCT系数数量，14*sampling_rate是块的数量
    x = x.permute(0, 1, 3, 4, 2) # [批次, 通道, 块数,块数 , 系数]
    x = x.view(bs, 3, 14 * 14 * sampling_rate * sampling_rate, 8, 8) # 重塑为[批次, 通道, 总块数, 块高, 块宽]
    x = dct.block_idct(x) # 将频率域DCT系数转换回空间域，每个8×8的DCT系数块变回像素块
    x = x.view(bs * 3, 14 * 14 * sampling_rate * sampling_rate, 64) # [批次×通道, 块数, 块内像素数]
    x = x.transpose(1, 2) # [批次×通道, 64, 块数]
    # 使用fold操作将块重新组合成完整图像，unfold的逆操作，重建图像的尺寸（896×896）
    x = F.fold(x, output_size=(112 * sampling_rate, 112 * sampling_rate),
               kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.view(bs, 3, 112 * sampling_rate, 112 * sampling_rate) # [批次, 3通道, 896, 896]
    x += 128
    x = dct.to_rgb(x)
    x /= 255
    # 下采样
    x = F.interpolate(x, scale_factor=1 / sampling_rate, mode='bilinear', align_corners=True)
    x = x.clamp(min=0.0, max=1.0)
    return x
