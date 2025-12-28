import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# 把连续的latent向量映射到离散的codebook中（embedding）
class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e # number of embeddings
        self.e_dim = e_dim # dimension of each embedding
        self.beta = beta # commitment loss parameter
        self.legacy = legacy # whether to use the legacy loss or not

        self.embedding = nn.Embedding(self.n_e, self.e_dim) # embedding layer
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            # load remapping indices， remap should be a path to a npy file，name as used
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            # numbers of used embeddings
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                # use an extra embedding for all unknown indices
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        # whether to return indices in (B, H, W) shape or (B, H*W, 1) shape
        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        # expect the shape of inds is more than one,like (B, N) or (B, H, W)
        ishape = inds.shape
        assert len(ishape) > 1
        # flatten to (B, L) L=H*W or others
        inds = inds.reshape(ishape[0], -1)
        # move used to the device of inds
        used = self.used.to(inds)
        # 比较每个inds元素是否等于used中的某个元素，得到一个(B, L, len(used))的布尔矩阵 match[b,i,j]=1 当且仅当 inds[b,i]==used[j]
        match = (inds[:, :, None] == used[None, None, ...]).long()
        # 每个inds元素在used中的位置，没有找到则全0
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        """
        import torch
        used = torch.tensor([5, 9, 42])           # num_used = 3
        inds = torch.tensor([[9, 5, 7], [42, 42, 1]])  # shape (2,3) batch=2, L=3
        # 模拟 match 计算
        match = (inds[:, :, None] == used[None, None, :]).long()
        # match 的值：
        # batch 0: inds [9,5,7] 对应 match rows -> [[0,1,0],[1,0,0],[0,0,0]]
        # batch 1: inds [42,42,1] -> [[0,0,1],[0,0,1],[0,0,0]]

        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        # new 初始是: tensor([[1,0,0],[2,2,0]])
        # unknown 是: tensor([[False, False, True],[False, False, True]])
        # 对 unknown 做处理后（例如 unknown -> random in [0, re_embed)):
        # new 的最后一列会被替换为随机或 extra index

        """
        # 对于没有找到的元素，根据unknown_index设置为随机值或extra index
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        # reshape back to original shape
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        # 输入 inds 在这里是假定属于 used 子集的 indices（范围在 0..re_embed-1），目的是把它们映回原来的全局索引空间（即 used 数组中对应的值）
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds) #等价于used[None,:].repeat(batch,1),对每一行，用 inds 指定的 col-index 从 used 的行中“gather”对应值，得到 shape (batch, L)，即把相对索引转换为原始 index 值
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten channel should be equal to e_dim
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        # 全部空间位置展开为第一维 N = batch * h * w，得到 (N, e_dim)。这便于做与 n_e 个 embedding 的距离比较（每行代表一个空间位置的向量）
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        """
        z_flattened: (N, e_dim),N = batch*h*w

        self.embedding.weight: (n_e, e_dim)

        torch.sum(z_flattened ** 2, dim=1, keepdim=True) -> (N, 1)

        torch.sum(self.embedding.weight ** 2, dim=1) -> (n_e,)（会广播到 (N, n_e)）。

        rearrange(self.embedding.weight, 'n d -> d n') -> (e_dim, n_e)

        torch.einsum('bd,dn->bn', z_flattened, ...) -> (N, n_e)，等价于 z_flattened @ embedding_weight.T.

        最终 d shape (N, n_e)，每个元素 d[i, j] 是 ||z_i - e_j||^2。
        """

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1) # (N,) 每个位置选择距离最近的 embedding 的 index
        z_q = self.embedding(min_encoding_indices).view(z.shape) # 先得到 (N, e_dim)，再 reshape 回 (b, h, w, c)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    # 给定 indices（一维或 (batch, h, w) 等），返回对应的 embedding 向量，并可 reshape 成 (batch, c, h, w) 形式
    # 如果使用 remap：先 unmap_to_all 把局部索引映回全局索引，然后从 embedding 中取值
    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q