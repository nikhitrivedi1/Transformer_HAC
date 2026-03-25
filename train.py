# Main Training Script

# Libraries
import torch
import torch.nn as nn
import tqdm
import argparse
import yaml
import json
import gzip
import os
import signal
import sys
import numpy as np
from torch.utils.data import DataLoader
from data.c24_data import C24_Dataset
from models.patch_tst import PatchTST
from models.cnn_model import CNNModel
from torchmetrics.classification import MulticlassF1Score

# ==============================================================================
# GLOBAL STATE FOR SIGNAL HANDLING
# ==============================================================================
CHECKPOINT_REQUESTED = False
CHECKPOINT_STATE = {}  # Will hold references to model, optimizer, scheduler, epoch, config


def handle_usr1_signal(signum, frame):
    """Handle USR1 signal - set flag to save checkpoint at next safe point."""
    global CHECKPOINT_REQUESTED
    print("\n" + "="*60)
    print("USR1 SIGNAL RECEIVED - Will save checkpoint and exit")
    print("="*60)
    CHECKPOINT_REQUESTED = True


def save_checkpoint(checkpoint_dir, run_name, model, optimizer, scheduler, epoch, config,
                    best_eval_loss=None, epochs_no_improve=0):
    """Save model checkpoint for resuming later (only called on USR1 signal)."""
    checkpoint_path = os.path.join(checkpoint_dir, run_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'config': config,
        'best_eval_loss': best_eval_loss,
        'epochs_no_improve': epochs_no_improve,
    }
    
    filepath = os.path.join(checkpoint_path, 'checkpoint.pt')
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to: {filepath}")
    print(f"Training can resume from epoch {epoch + 1}")
    return filepath


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load model checkpoint for resuming training."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    best_eval_loss = checkpoint.get('best_eval_loss', None)
    epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
    
    print(f"Resuming from epoch {start_epoch}")
    if best_eval_loss is not None:
        print(f"Best eval loss so far: {best_eval_loss:.4f}, epochs without improvement: {epochs_no_improve}")
    
    return start_epoch, best_eval_loss, epochs_no_improve

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


def train_model(model, train_loader, eval_loader, config, start_epoch=0, resume_checkpoint=None):
    global CHECKPOINT_REQUESTED, CHECKPOINT_STATE
    
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
        loss_function = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss function {config['loss_function']} not supported")
    # Initialize the scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    # Initialize the device
    model.to(device)
    
    # Early stopping configuration (based on eval loss - lower is better)
    # Uses validation loss if run_type='validation', test loss if run_type='test'
    patience = config.get('early_stopping_patience', 5)
    best_eval_loss = None
    epochs_no_improve = 0
    
    # Load checkpoint if resuming
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        start_epoch, best_eval_loss, epochs_no_improve = load_checkpoint(
            resume_checkpoint, model, optimizer, scheduler, device
        )
    
    # Store references for signal handler
    CHECKPOINT_STATE['model'] = model
    CHECKPOINT_STATE['optimizer'] = optimizer
    CHECKPOINT_STATE['scheduler'] = scheduler
    CHECKPOINT_STATE['config'] = config

    for epoch in tqdm.tqdm(range(start_epoch, config['epochs']), desc="Training", initial=start_epoch, total=config['epochs']):
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

        # Compute current eval loss (validation or test depending on run_type)
        current_eval_loss = b_eval_loss / len(eval_loader)        
        loss_type = 'valid' if config['run_type'] == 'validation' else 'test'
        
        # Handle both Validation and Test Sets here
        if config['run_type'] == 'validation':
            model.run.log({
                'train/epoch_loss': b_train_loss / len(train_loader),
                'valid/epoch_loss': current_eval_loss,
                'lr': scheduler.get_last_lr()[0],
                'f1_score': f1_score.compute(),
            }, 
            step = epoch)
        elif config['run_type'] == 'test': 
            model.run.log({
                'train/epoch_loss': b_train_loss / len(train_loader),
                'test/epoch_loss': current_eval_loss,
                'lr': scheduler.get_last_lr()[0],
                'f1_score': f1_score.compute(),
            }, 
            step = epoch)
        
        # Early stopping check (based on validation/test loss - lower is better)
        if best_eval_loss is None or current_eval_loss < best_eval_loss:
            best_eval_loss = current_eval_loss
            epochs_no_improve = 0
            print(f"  [BEST] New best {loss_type} loss: {best_eval_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  [EARLY STOP] No improvement for {epochs_no_improve}/{patience} epochs (best {loss_type} loss: {best_eval_loss:.4f})")
        
        if epochs_no_improve >= patience:
            print(f"\n{'='*60}")
            print(f"EARLY STOPPING: No improvement for {patience} consecutive epochs")
            print(f"Best {loss_type} loss: {best_eval_loss:.4f}")
            print(f"{'='*60}")
            break  # Just stop training - no weights saved
        
        # Check if checkpoint was requested (USR1 signal received)
        # This is the ONLY place weights are saved (on interruption)
        if CHECKPOINT_REQUESTED:
            checkpoint_dir = config.get('checkpoint_dir', 'weight_checkpoints')
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                run_name=config['run_name'],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=config,
                best_eval_loss=best_eval_loss,
                epochs_no_improve=epochs_no_improve
            )
            print("Exiting due to checkpoint request...")
            sys.exit(0)  # Exit cleanly - sweep will resubmit
            
    return model



def main():
    DATA_DIR = 'capture24/final_data_1024_mode_v'

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
    parser.add_argument(
        '--positional_encoding',
        type=str,
        default='fixed',
        choices=['fixed', 'learned', 'rope'],
        help='Position encoding: fixed (sinusoidal), learned (parameter), or rope (rotary in attention)',
    )
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--classifier', type=str, default='linear')
    parser.add_argument('--jitter_prob', type=float, default=0.5)
    parser.add_argument('--shift_prob', type=float, default=0.2)
    parser.add_argument('--twarp_prob', type=float, default=0.3)
    parser.add_argument('--mwarp_prob', type=float, default=0.3)
    # Checkpointing arguments
    parser.add_argument('--checkpoint_dir', type=str, default='weight_checkpoints',
                        help='Directory to save weight checkpoints on interruption')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Number of epochs without improvement before stopping (default: 5)')

    args = parser.parse_args()
    
    # Set up signal handler for graceful checkpoint on timeout
    signal.signal(signal.SIGUSR1, handle_usr1_signal)
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
    config['run_type'] = args.run_type
    config['output_dir'] = args.output_dir
    config['classifier'] = args.classifier
    config['jitter_prob'] = args.jitter_prob
    config['shift_prob'] = args.shift_prob
    config['twarp_prob'] = args.twarp_prob
    config['mwarp_prob'] = args.mwarp_prob
    config['checkpoint_dir'] = args.checkpoint_dir
    config['resume_from'] = args.resume_from
    config['early_stopping_patience'] = args.early_stopping_patience
    config['positional_encoding'] = args.positional_encoding
    print(f"positional_encoding: {config['positional_encoding']}")
    print(f"Early stopping patience: {config['early_stopping_patience']} epochs")
    if args.resume_from:
        print(f"Will resume from checkpoint: {args.resume_from}")
    # EXCLUDE_CLASSES = ['sleep', 'sitting']
    EXCLUDE_CLASSES = None

    # Based on args - load the correct data files
    # Load the data
    with gzip.open(f'{DATA_DIR}/X_train.npy.gz', 'rb') as f:
        X_train = np.load(f)
    with gzip.open(f'{DATA_DIR}/Y_train.npy.gz', 'rb') as f:
        Y_train = np.load(f)

    # Load the index to label and label to index
    with open(f'{DATA_DIR}/label_to_index.json', 'r') as f:
        data = json.load(f)

    idx_to_label = data['index_to_label']
    label_to_idx = data['label_to_index']

    if config['run_type'] == 'validation':
        with gzip.open(f'{DATA_DIR}/X_valid.npy.gz', 'rb') as f:
            X_valid = np.load(f)
        with gzip.open(f'{DATA_DIR}/Y_valid.npy.gz', 'rb') as f:
            Y_valid = np.load(f)
        eval_dataset = C24_Dataset(X_valid, Y_valid, idx_to_label, label_to_idx, exclude_classes=EXCLUDE_CLASSES)
        print(eval_dataset.get_class_distribution())
    elif config['run_type'] == 'test':
        with gzip.open(f'{DATA_DIR}/X_test.npy.gz', 'rb') as f:
            X_test = np.load(f)
        with gzip.open(f'{DATA_DIR}/Y_test.npy.gz', 'rb') as f:
            Y_test = np.load(f)
        eval_dataset = C24_Dataset(X_test, Y_test, idx_to_label, label_to_idx, exclude_classes=EXCLUDE_CLASSES)
        print(eval_dataset.get_class_distribution())


    # Create the test and train datasests
    augmentation_args = {
        'jitter_sigma': 0.1, 'jitter_prob': config['jitter_prob'],
        'shift_window': 200, 'shift_prob': config['shift_prob'],
        'twarp_sigma': 0.05, 'twarp_knots': 4, 'twarp_prob': config['twarp_prob'],
        'mwarp_sigma': 0.05, 'mwarp_knots': 2, 'mwarp_prob': config['mwarp_prob'],
    }
    train_dataset = C24_Dataset(X_train, Y_train, idx_to_label, label_to_idx, exclude_classes=EXCLUDE_CLASSES, augs_args=augmentation_args)
    print(train_dataset.get_class_distribution())


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
        model = train_model(
            model, 
            train_loader, 
            valid_loader, 
            config,
            start_epoch=0,
            resume_checkpoint=config['resume_from']
        )
    finally:
        model.run.finish()

    # Save the model - don't need to save the model as we are using WB for validation - however, for test we do need to save the model
    # if config['run_type'] == 'test':
    #     torch.save(model.state_dict(), f'{path_folder}/patchtst_model.pth')
    #     # Save the config
    #     with open(f'{path_folder}/config.yaml', 'w') as f:
    #         yaml.dump(config, f)
    # else:
    #     pass
    

if __name__ == "__main__":
    main()
