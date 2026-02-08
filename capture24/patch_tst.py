# PatchTST Based HAC Detection
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from capture24.custom_encoder import CustomEncoder
import math
# import weights and biases for logging
import wandb



class PatchingLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.patch_length = config['patch_length']
        self.stride = config['stride']

    def forward(self, x: torch.Tensor):
        # X will be the shape (B, T, C -> B, C, T)
        x = x.transpose(1, 2)
        # add padding before unfold
        if (x.shape[-1] - self.patch_length) % self.stride == 0:
            pad_len = self.stride
        else:
            pad_len = self.stride - (x.shape[-1] - self.patch_length) % self.stride
            print(f"Padding length partial: {pad_len}")

        x = F.pad(x, (0,pad_len), mode='replicate')
        x = x.transpose(1, 2)
        patches = x.unfold(1, self.patch_length, self.stride)
        return patches

class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.pos_encodings = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # sin and cos for even and odd indices respectively
        self.pos_encodings[:, :, 0::2] = torch.sin(position * div_term) # even indices
        self.pos_encodings[:, :, 1::2] = torch.cos(position * div_term) # odd indices
    
    def forward(self, x):
        return self.pos_encodings[:, :x.shape[1], :].to(x.device)

class PatchTST(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.inst_norm = nn.InstanceNorm1d(config['num_channels'])
        self.num_batches = config['num_batches']

        self.num_patches = ((config['lookback_window'] - config['patch_length']) // config['stride']) + 2
        self.patching_layer = PatchingLayer(config)

        if config['channel_independence']:
            self.flatten1 = nn.Flatten(0,1)
            self.linear1 = nn.Linear(config['patch_length'], config['d_model'])
        else:
            self.flatten1 = nn.Flatten(2,3)
            self.linear1 = nn.Linear(config['patch_length']*config['num_channels'], config['d_model'])

        # trainable conditional encoding
        if config['trainable_pos_encoding']:
            self.pos_encoding = nn.Parameter(torch.zeros(1, self.num_patches, config['d_model']))
        else:
            self.pos_encoding = FixedPositionalEncoding(config['d_model'], self.num_patches)

        # Transformer Encoder
        # Custom Encoder used for attention studies
        # encoder_layer = CustomEncoder(
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['t_heads'], 
            dim_feedforward=config['t_dim_feedforward'], 
            dropout=config['t_dropout'],
            batch_first=True,
            activation=config['t_activation'],
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config['t_num_layers'])

        # hook for attention maps - only needed for testing
        if config['hook_attention_maps']:
            self.attention_maps = []
            def hook_attention_maps(module, input, output):
                self.attention_maps.append(output[1].detach())
            for layer in self.encoder.layers:
                layer.self_attn.register_forward_hook(hook_attention_maps)
                # will output a tensor of shape [B, N, N] -> where B represents the encoder layer

        # Flatten the number of patches and the dimension of the model
        self.flatten2 = nn.Flatten(1, 2)


        # Classifier Layer - two options - linear or MLP
        if config['classifier'] == 'linear':
            self.classifier = nn.Linear(self.num_patches*config['d_model'], config['num_classes'])
        elif config['classifier'] == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(self.num_patches*config['d_model'], config['mlp_hidden_size']),
                nn.ReLU(),
                nn.Linear(config['mlp_hidden_size'], config['num_classes']),
            )
        else:
            raise ValueError(f"Classifier {config['classifier']} not supported")

        self.config = config

        # WB Run - for logging
        self.run = wandb.init(
            entity="nikhitrivedi1-northeastern-university",
            project="PatchTST_baseline",
            config=config,
            name=config['run_name'],
        )

    def forward(self, x):
        # Nomenclature to match PatchTST paper
        # B = batch size, M = channels, N = number of patches, P = patch length, T = time steps
        x = x.permute(0, 2, 1) # input is (B, T, C) -> (B, C, T)
        x = self.inst_norm(x) # requires B, C, T

        x = x.permute(0, 2, 1) # output is (B, C, T) -> (B, T, C)

        x = self.patching_layer(x) # output is (B, N, C, P)

        if self.config['channel_independence']:
            x = x.permute(0,2,1,3) # output is (B, C, N, P)

        x = self.flatten1(x) # output is (B, N, P*C) if channel_independence is False, otherwise (B*N, P, C)

        x = self.linear1(x) # output is (B, N, d_model)

        x = x + self.pos_encoding.forward(x) # output is (B, N, d_model)
        x = self.encoder(x) # output is (B, N, d_model)

        if self.config['channel_independence']:
            BC, N, d_model = x.shape
            B = BC // self.config['num_channels']
            # Pool the channels 
            x = x.view(B, self.config['num_channels'], N, d_model)
            x = x.mean(dim=1)

        # Flatten all the patches together with embedding 
        x = self.flatten2(x) # output is (B, N*d_model) -> if channel independence is True, then (B, N * 
        x = self.classifier(x) # output is (B, num_classes) # final output shape
        return x