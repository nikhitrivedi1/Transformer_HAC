# PatchTST Based HAC Detection
import numpy as np
import json
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import yaml
import tqdm
import matplotlib.pyplot as plt
import gzip
import argparse
# from custom_encoder import CustomEncoder
import os


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
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.num_patches, config['d_model']))

        # Transformer Encoder
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

        # self.classifier = nn.Linear(self.num_patches*config['t_dim_feedforward'], config['num_classes'])

        self.classifier = nn.Sequential(
            nn.Linear(self.num_patches*config['d_model'], config['mlp_hidden_size']),
            nn.ReLU(),
            nn.Linear(config['mlp_hidden_size'], config['num_classes']),
        )

        self.config = config

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

        x = x + self.pos_encoding # output is (B, N, d_model)
        x = self.encoder(x) # output is (B, N, d_model)


        if self.config['channel_independence']:
            BC, N, d_model = x.shape
            B = BC // self.config['num_channels']
            # Pool the channels 
            x = x.view(B, self.config['num_channels'], N, d_model)
            x = x.mean(dim=1)

        # Flatten all the patches together with embedding 
        x = self.flatten2(x) # output is (B, N*d_model)
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


def train_model(model, train_loader, test_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Evaluation Metrics
    # accuracy = Accuracy(task='multiclass', num_classes=config['num_classes'])
    # f1_score = F1Score(task='multiclass', num_classes=config['num_classes'])
    # precision = Precision(task='multiclass', num_classes=config['num_classes'])
    # recall = Recall(task='multiclass', num_classes=config['num_classes'])
    # Initialize the optimizer
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['lr'],
            )
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['lr'],
            weight_decay=config['weight_decay'],
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
    elif config['scheduler'] == 'none':
        scheduler = None
    else:
        raise ValueError(f"Scheduler {config['scheduler']} not supported")

    # Initialize the loss function
    if config['loss'] == 'cross_entropy':
        loss_function = nn.CrossEntropyLoss()
    elif config['loss'] == 'weighted_cross_entropy':
        loss_function = nn.CrossEntropyLoss(weight=config['weight'])
    else:
        raise ValueError(f"Loss function {config['loss_function']} not supported")
    # Initialize the scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    # Initialize the device
    model.to(device)

    # Train the model
    e_train_losses = []
    e_test_losses = []
    b_train_losses = []
    b_test_losses = []
    lr_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []
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
            b_train_losses.append(loss.item())

        if scheduler is not None:
            lr_scores.append(scheduler.get_last_lr()[0])
            scheduler.step()

        model.eval() # eval mode
        with torch.no_grad():
            b_test_loss = 0
            for x_batch, y_batch in tqdm.tqdm(test_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(x_batch)

                loss = loss_function(output, y_batch)
                b_test_loss += loss.item()
                b_test_losses.append(loss.item())

                # come up with class prediction
                # class_prediction = torch.softmax(output, dim=1)
                # class_prediction = torch.argmax(class_prediction, dim=1)

                # # Evaluation Metrics per batch
                # accuracy_scores.append(accuracy(class_prediction.cpu(), y_batch.cpu()))
                # precision_scores.append(precision(class_prediction.cpu(), y_batch.cpu()))
                # recall_scores.append(recall(class_prediction.cpu(), y_batch.cpu()))
                # f1_scores.append(f1_score(class_prediction.cpu(), y_batch.cpu()))


        e_train_losses.append(b_train_loss / len(train_loader))
        e_test_losses.append(b_test_loss / len(test_loader))

    
    losses = {
        'b_train_losses': b_train_losses,
        'b_test_losses': b_test_losses,
        'e_train_losses': e_train_losses,
        'e_test_losses': e_test_losses,
        'lr_scores': lr_scores,
    }

    metrics = {
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'f1_scores': f1_scores,
        'accuracy_scores': accuracy_scores,
    }
    return model, losses, metrics


def main():
    # Load the data
    with gzip.open('final_data_1024/X_train.npy.gz', 'rb') as f:
        X_train = np.load(f)
    with gzip.open('final_data_1024/Y_train.npy.gz', 'rb') as f:
        Y_train = np.load(f)
    with gzip.open('final_data_1024/X_test.npy.gz', 'rb') as f:
        X_test = np.load(f)
    with gzip.open('final_data_1024/Y_test.npy.gz', 'rb') as f:
        Y_test = np.load(f)

    # Load the index to label and label to index
    with open('final_data_1024/label_to_index.json', 'r') as f:
        data = json.load(f)

    idx_to_label = data['index_to_label']
    label_to_idx = data['label_to_index']

    # Create the test and train datasests
    train_dataset = C24_Dataset(X_train, Y_train, idx_to_label, label_to_idx)
    test_dataset = C24_Dataset(X_test, Y_test, idx_to_label, label_to_idx)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_1.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

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
    test_loader = DataLoader(
        test_dataset, 
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


    # Calculate the weight for the weighted cross entropy loss
    # counts = torch.bincount(torch.from_numpy(Y_train), minlength=config['num_classes'])
    # weights = 1 / counts
    # weights = weights * ( config['num_classes'] / counts.sum())
    # config['weight'] = weights
    # print(f"Weight: {weights}")
    # exit()


    # Train the model
    model, losses, metrics = train_model(model, train_loader, test_loader, config)

    # Save the model
    torch.save(model.state_dict(), f'{path_folder}/patchtst_model.pth')

    # Save the losses
    np.save(f'{path_folder}/b_train_losses.npy', losses['b_train_losses'])
    np.save(f'{path_folder}/b_test_losses.npy', losses['b_test_losses'])
    np.save(f'{path_folder}/e_train_losses.npy', losses['e_train_losses'])
    np.save(f'{path_folder}/e_test_losses.npy', losses['e_test_losses'])

    # Save the metrics
    np.save(f'{path_folder}/precision_scores.npy', metrics['precision_scores'])
    np.save(f'{path_folder}/recall_scores.npy', metrics['recall_scores'])
    np.save(f'{path_folder}/f1_scores.npy', metrics['f1_scores'])
    np.save(f'{path_folder}/accuracy_scores.npy', metrics['accuracy_scores'])

if __name__ == "__main__":
    main()