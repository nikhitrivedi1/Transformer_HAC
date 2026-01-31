# PatchTST Based HAC Detection
import numpy as np
import json
import torch
from torch import nn
from torchmetrics.classification import MulticlassF1Score
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import yaml
import tqdm
import matplotlib.pyplot as plt
import gzip
import argparse
# from custom_encoder import CustomEncoder
import os
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
            project="PatchTST_baseline_patch_study",
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

class C24_Dataset(Dataset):
    def __init__(self, X, Y, idx_to_label, label_to_idx):
        # convert from numpy to torch tensors
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()
        self.idx_to_label = idx_to_label
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # X will return a tensor of shape [T, C] and Y will return a tensor of shape [1]
        return self.X[index], self.Y[index]



def build_param_groups_adamw(model, weight_decay):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Rules commonly used for Transformers
        if name.endswith("bias"):
            no_decay.append(param)
        elif "norm" in name.lower() or "layernorm" in name.lower():
            no_decay.append(param)
        elif "embedding" in name.lower() or "embed" in name.lower():
            no_decay.append(param)  # optional but common
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]



def train_model(model, train_loader, eval_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")


    # Initialize the optimizer
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['lr'],
            )
    elif config['optimizer'] == 'adamw':
        # Need to configure seperate groups -> do not want weight_decay value to apply to bias or layeer norm
        param_group = build_param_groups_adamw(model, config['weight_decay'])
        optimizer = torch.optim.AdamW(
            param_group,
            lr = config['lr'],
            eps = 1e-10
        )
    elif config['optimizer'] == 'sgdwr':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            momentum=config['momentum'],
        )
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")


    # Scheduler
    if config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    elif config['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    elif config['scheduler'] == 'cosine_restart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=config['min_lr'])
    elif config['scheduler'] == 'cosine_warmup':
        # LinearLR for the first portion of the training
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=.00001, # 1e-5
            end_factor=1.0, 
            total_iters=config['warmup_epochs']
        )
        # CosineAnnealingLR for the remaining time
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['epochs'] - config['warmup_epochs'], 
            eta_min=config['min_lr']
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            [warmup, cosine],
            milestones=[config['warmup_epochs']]
        )

    elif config['scheduler'] == 'none':
        scheduler = None
    else:
        raise ValueError(f"Scheduler {config['scheduler']} not supported")

    # Initialize the loss function
    if config['loss'] == 'cross_entropy':
        loss_function = nn.CrossEntropyLoss()
    elif config['loss'] == 'w_cross_entropy':
        # calculate the weights
        loss_function = nn.CrossEntropyLoss(weight=config['weight'].to(device))
    else:
        raise ValueError(f"Loss function {config['loss_function']} not supported")
    # Initialize the scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    # Initialize the device
    model.to(device)

    for epoch in tqdm.tqdm(range(config['epochs']), desc="Training"):
        model.train() # train mode
        b_train_loss = 0
        for x_batch, y_batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()

            b_train_loss += loss.item()

        if scheduler is not None:
            # lr_scores.append(scheduler.get_last_lr()[0])
            scheduler.step()

        model.eval() # eval mode
        with torch.no_grad():
            f1_score = MulticlassF1Score(num_classes=10, average='macro').to(device)

            b_eval_loss = 0
            for x_batch, y_batch in tqdm.tqdm(eval_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(x_batch)

                loss = loss_function(output, y_batch)
                b_eval_loss += loss.item()

                # calcualte f1_score 
                preds = output.argmax(dim=1)
                f1_score.update(preds, y_batch)

        # Handle both Validation and Test Sets here
        if config['run_type'] == 'validation':
            model.run.log({
                'train/epoch_loss': b_train_loss / len(train_loader),
                'valid/epoch_loss': b_eval_loss / len(eval_loader),
                'lr': scheduler.get_last_lr()[0],
                'f1_score': f1_score.compute(),
            }, 
            step = epoch)
        elif config['run_type'] == 'test': 
            model.run.log({
                'train/epoch_loss': b_train_loss / len(train_loader),
                'test/epoch_loss': b_eval_loss / len(eval_loader),
                'lr': scheduler.get_last_lr()[0],
                'f1_score': f1_score.compute(),
            }, 
            step = epoch)
    return model


def main():

    # Import Config from SLURM Job Array
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_1.yaml')
    # Search Space Parameters
    parser.add_argument('--patch_length', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--t_dropout', type=float, default=0.25)
    parser.add_argument('--t_num_layers', type=int, default=1)
    parser.add_argument('--run_name', type=str, default='config_1')
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--run_type', type=str, default='validation')
    parser.add_argument("--trainable_pos_encoding", action="store_true")
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--classifier', type=str, default='linear')

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    config['patch_length'] = args.patch_length
    config['stride'] = args.stride
    config['lr'] = args.lr
    config['t_dropout'] = args.t_dropout
    config['t_num_layers'] = args.t_num_layers
    config['run_name'] = args.run_name
    config['t_heads'] = args.n_heads
    config['optimizer'] = args.optimizer_type
    config['weight_decay'] = args.weight_decay
    config['loss'] = args.loss
    config['weight'] = torch.from_numpy(np.load('final_data_1024_mode_v/weights.npy')).float()
    config['run_type'] = args.run_type
    config['output_dir'] = args.output_dir
    config['trainable_pos_encoding'] = args.trainable_pos_encoding
    config['classifier'] = args.classifier
    
    print(f"Trainable Pos Encoding: {config['trainable_pos_encoding']}")

    # Based on args - load the correct data files
    # Load the data
    with gzip.open('final_data_1024_mode_v/X_train.npy.gz', 'rb') as f:
        X_train = np.load(f)
    with gzip.open('final_data_1024_mode_v/Y_train.npy.gz', 'rb') as f:
        Y_train = np.load(f)

    # Load the index to label and label to index
    with open('final_data_1024_mode_v/label_to_index.json', 'r') as f:
        data = json.load(f)

    idx_to_label = data['index_to_label']
    label_to_idx = data['label_to_index']

    if config['run_type'] == 'validation':
        with gzip.open('final_data_1024_mode_v/X_valid.npy.gz', 'rb') as f:
            X_valid = np.load(f)
        with gzip.open('final_data_1024_mode_v/Y_valid.npy.gz', 'rb') as f:
            Y_valid = np.load(f)
        eval_dataset = C24_Dataset(X_valid, Y_valid, idx_to_label, label_to_idx)
    elif config['run_type'] == 'test':
        with gzip.open('final_data_1024_mode_v/X_test.npy.gz', 'rb') as f:
            X_test = np.load(f)
        with gzip.open('final_data_1024_mode_v/Y_test.npy.gz', 'rb') as f:
            Y_test = np.load(f)
        eval_dataset = C24_Dataset(X_test, Y_test, idx_to_label, label_to_idx)


    # Create the test and train datasests
    train_dataset = C24_Dataset(X_train, Y_train, idx_to_label, label_to_idx)

    if config['output_dir'] is not None:
        path_folder = config['output_dir']
    else:
        # Create a path folder for this version of the model
        path_folder = f"{args.config.split('_')[-1].split('.')[0]}"

    
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['num_batches'], 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False, 
        drop_last=True, 
        persistent_workers=False, 
        prefetch_factor=None,
    )
    valid_loader = DataLoader(
        eval_dataset, 
        batch_size=config['num_batches'], 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False, 
        drop_last=False, 
        persistent_workers=False, 
        prefetch_factor=None,
    )

    # Initialize the model
    model = PatchTST(config)

    # Train the model
    try:
        model = train_model(model, train_loader, valid_loader, config)
    finally:
        model.run.finish()

    # Save the model - don't need to save the model as we are using WB for validation - however, for test we do need to save the model
    if config['run_type'] == 'test':
        torch.save(model.state_dict(), f'{path_folder}/patchtst_model.pth')
        # Save the config
        with open(f'{path_folder}/config.yaml', 'w') as f:
            yaml.dump(config, f)
    else:
        pass
    

if __name__ == "__main__":
    main()