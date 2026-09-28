
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.scalarquantizer import ScalarQuantizer
# from models.vectorquantizer import VectorQuantizer
from models.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, m, save_img_embedding_map=False,
                 method='uniform', B=4):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)

        # pass continuous latent vector through discretization bottleneck
        self.scalar_quantization = ScalarQuantizer(
            n_embeddings, int(embedding_dim/m), beta, method=method, B=B)
        # self.vector_quantization = VectorQuantizer(
        #     n_embeddings, int(embedding_dim/m), beta)

        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None
        self.tanh = nn.Tanh()

    def encode(self, x):
        # 이미지 -> 연속 latent z (tanh로 [-1, 1] 범위), shape (N, C, H, W)
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        return self.tanh(z_e)

    def forward(self, x, verbose=False):

        z_e = self.encode(x)

        embedding_loss, z_q, perplexity, e_mean, Bits = self.scalar_quantization(
            z_e)
        # embedding_loss, z_q, perplexity, e_mean, Bits = self.vector_quantization(
        #     z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, e_mean, Bits
