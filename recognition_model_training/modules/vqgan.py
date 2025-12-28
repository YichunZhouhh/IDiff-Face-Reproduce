import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
from quantization import VectorQuantizer2 as VectorQuantizer
from modules import Encoder, Decoder


class VQEncoderInterface(torch.nn.Module):

    def __init__(self,
                 encoder_state_dict_path: str = '/remote-home/share/yxmi/ckpts/frcsyn_autoencoder/first_stage_encoder_state_dict.pt'):
        super().__init__()

        embed_dim = 3

        self.encoder = Encoder(double_z=False, z_channels=3, resolution=256, in_channels=3, out_ch=3, ch=128,
                               ch_mult=(1, 2, 4), num_res_blocks=2, attn_resolutions=[], dropout=0.0)
        self.quant_conv = torch.nn.Conv2d(3, embed_dim, 1)

        if encoder_state_dict_path is not None:
            state_dict = torch.load(encoder_state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h


class VQDecoderInterface(torch.nn.Module):

    def __init__(self,
                 decoder_state_dict_path: str = '/remote-home/share/yxmi/ckpts/frcsyn_autoencoder/first_stage_decoder_state_dict.pt'):
        super().__init__()

        embed_dim = 3
        n_embed = 8192

        self.decoder = Decoder(double_z=False, z_channels=3, resolution=256, in_channels=3, out_ch=3, ch=128,
                               ch_mult=(1, 2, 4), num_res_blocks=2, attn_resolutions=[], dropout=0.0)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, 3, 1)

        if decoder_state_dict_path is not None:
            state_dict = torch.load(decoder_state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class BufferAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_conv = torch.nn.Sequential(torch.nn.Conv2d(9, 3, 1),
                                              torch.nn.BatchNorm2d(3),
                                              torch.nn.Tanh())  # [-1, 1]
        # self.embed_conv = torch.nn.Sequential(torch.nn.Conv2d(9, 3, 1))
        self.embed_deconv = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 9, 1),
                                                torch.nn.Sigmoid())  # [0, 1]
        self.encoder = VQEncoderInterface()
        self.decoder = VQDecoderInterface()
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def sample_from_latent(self, batch_size=1, shape=(32, 32)):
        # Step 1: 获取代码本体的大小
        codebook_size = self.decoder.quantize.n_e  # 获取代码本体的大小

        # Step 2: 随机从代码本体中采样索引
        sample_shape = (batch_size, *shape)
        sampled_indices = torch.randint(0, codebook_size, sample_shape)  # 随机采样索引，sample_shape = (batch_size, H, W)

        # Step 3: 根据索引从代码本体中获取对应的向量
        sampled_latents = self.decoder.quantize.embedding.weight[sampled_indices]  # 查找对应的量化向量
        sampled_latents = sampled_latents.permute(0, 3, 1, 2).contiguous()  # 调整维度为 (batch_size, C, H, W)，与卷积操作兼容

        # Step 4: 通过后量化卷积层并生成图像
        quant = self.decoder.post_quant_conv(sampled_latents)  # 通过量化后的卷积层
        generated = self.decoder.decoder(quant)  # 通过解码器生成图像

        return generated

    def forward(self, x):
        self.encoder.eval()
        self.decoder.eval()

        x = self.embed_conv(x)
        x_latent = self.encoder(x)
        x_recon = self.decoder(x_latent)
        x_deconv = self.embed_deconv(x_recon)
        return x, x_recon, x_deconv


if __name__ == '__main__':
    model = BufferAE()
    x = torch.randn(1, 9, 128, 128)
    print(x.min(), x.max())
    embed = model.embed_conv(x)
    print(embed.min(), embed.max())

    pass
    # encoder = Encoder(double_z=False, z_channels=3, resolution=256, in_channels=3, out_ch=3, ch=128, ch_mult=(1, 2, 4),
    #                   num_res_blocks=2, attn_resolutions=[], dropout=0.0)
    # decoder = Decoder(double_z=False, z_channels=3, resolution=256, in_channels=3, out_ch=3, ch=128, ch_mult=(1, 2, 4),
    #                   num_res_blocks=2, attn_resolutions=[], dropout=0.0)
    #
    # param_size = 0
    # for param in encoder.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in encoder.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    #
    # size_all_mb = (param_size + buffer_size) / 1024 ** 2
    # print('model size: {:.3f}MB'.format(size_all_mb))
    #
    # a = torch.rand(1, 3, 128, 128)
    # b = encoder(a)
    # print(b.shape)
    # print(decoder(b).shape)
