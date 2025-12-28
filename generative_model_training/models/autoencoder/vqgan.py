from omegaconf import OmegaConf

import torch
from models.autoencoder.quantization import VectorQuantizer2 as VectorQuantizer
from models.autoencoder.modules import Encoder, Decoder


class VQEncoderInterface(torch.nn.Module):

    def __init__(self, first_stage_config_path: str, encoder_state_dict_path: str):
        super().__init__()
        
        embed_dim = 3
        # 读取 YAML 配置文件（路径由调用者传入），通常该文件包含 params -> ddconfig（即第一阶段 autoencoder 的网络超参）
        config = OmegaConf.load(first_stage_config_path)
        # 把配置中 ddconfig 提取出来，dd_config 是一个字典/映射
        dd_config = config.params.ddconfig

        self.encoder = Encoder(**dd_config)
        # 1x1 卷积层，把编码器输出的通道数变换到 embed_dim
        self.quant_conv = torch.nn.Conv2d(dd_config["z_channels"], embed_dim, 1)
        # 加载预训练的编码器权重（路径由调用者传入）
        state_dict = torch.load(encoder_state_dict_path)
        self.load_state_dict(state_dict)

    def forward(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h


class VQDecoderInterface(torch.nn.Module):

    def __init__(self, first_stage_config_path: str, decoder_state_dict_path: str):
        super().__init__()

        embed_dim = 3
        n_embed = 8192

        config = OmegaConf.load(first_stage_config_path)
        # "/home/grebe/IDiff-Face//models/autoencoder/first_stage_config.yaml")
        dd_config = config.params.ddconfig

        self.decoder = Decoder(**dd_config)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, dd_config["z_channels"], 1)

        state_dict = torch.load(decoder_state_dict_path)
        # "/home/grebe/IDiff-Face//models/autoencoder/first_stage_decoder_state_dict.pt"
        self.load_state_dict(state_dict)

    def forward(self, h, force_not_quantize=False):
        # h is expected to be in shape (batch, embed_dim, height, width)
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h) # quant shape (b,embed_dim/c, h, w)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
