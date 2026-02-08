# CNN Model for Capture24 - Exact Mirror of Capture24 Paper
# Architecture from Supplementary Table 1
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool1d(nn.Module):
    """
    Anti-aliased downsampling as described in:
    "Making Convolutional Networks Shift-Invariant Again" (Zhang, 2019)
    """
    def __init__(self, channels, kernel_size=3, stride=2):
        super().__init__()
        self.stride = stride
        self.channels = channels
        
        # Create blur kernel (binomial filter)
        if kernel_size == 3:
            kernel = torch.tensor([1., 2., 1.])
        elif kernel_size == 5:
            kernel = torch.tensor([1., 4., 6., 4., 1.])
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}")
        
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1).repeat(channels, 1, 1)
        self.register_buffer('kernel', kernel)
        self.pad = kernel_size // 2
    
    def forward(self, x):
        # Apply blur filter with groups=channels (depthwise)
        x = F.pad(x, (self.pad, self.pad), mode='circular')
        x = F.conv1d(x, self.kernel, stride=self.stride, groups=self.channels)
        return x


class ResBlock(nn.Module):
    """
    Residual block with two conv layers.
    Conv -> BN -> ReLU -> Conv -> BN -> Add residual
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=1, padding='same', padding_mode='circular')
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, stride=1, padding='same', padding_mode='circular')
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out


class ConvResBlock(nn.Module):
    """
    Conv layer followed by 3 ResBlocks, then anti-aliased downsampling.
    Conv(k, n) -> BN -> ReLU -> 3 x ResBlock(k, n) -> BlurPool(downsample_factor)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample_factor=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding='same', padding_mode='circular')
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 3 x ResBlock
        self.resblock1 = ResBlock(out_channels, kernel_size)
        self.resblock2 = ResBlock(out_channels, kernel_size)
        self.resblock3 = ResBlock(out_channels, kernel_size)
        
        # Anti-aliased downsampling
        self.downsample = BlurPool1d(out_channels, kernel_size=3, stride=downsample_factor)
    
    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn(self.conv(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.downsample(x)
        return x


class CNNModel(nn.Module):
    """
    CNN architecture from Capture24 paper (Supplementary Table 1).
    
    Architecture:
        (*, 3, 1000) -> Conv(3, 128) / 2               -> (*, 128, 500)
        (*, 128, 500) -> Conv(3, 128), 3xResBlock / 2  -> (*, 128, 250)
        (*, 128, 250) -> Conv(3, 256), 3xResBlock / 2  -> (*, 256, 125)
        (*, 256, 125) -> Conv(3, 256), 3xResBlock / 5  -> (*, 256, 25)
        (*, 256, 25)  -> Conv(3, 512), 3xResBlock / 5  -> (*, 512, 5)
        (*, 512, 5)   -> Conv(3, 512), 3xResBlock / 5  -> (*, 512, 1)
        (*, 512, 1)   -> Drop(0.5), FC(1024)           -> (*, 1024)
        (*, 1024)     -> Linear(6)                     -> (*, 6)
    """
    def __init__(self, num_channels=3, num_classes=6):
        super().__init__()
        
        # Layer 1: Conv(3, 128) / 2
        # Just conv + bn + relu + downsample (no ResBlocks)
        self.conv1 = nn.Conv1d(num_channels, 128, kernel_size=3, stride=1, padding='same', padding_mode='circular')
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.downsample1 = BlurPool1d(128, kernel_size=3, stride=2)
        
        # Layer 2: Conv(3, 128), 3 x ResBlock(3, 128) / 2
        self.layer2 = ConvResBlock(in_channels=128, out_channels=128, kernel_size=3, downsample_factor=2)
        
        # Layer 3: Conv(3, 256), 3 x ResBlock(3, 256) / 2
        self.layer3 = ConvResBlock(in_channels=128, out_channels=256, kernel_size=3, downsample_factor=2)
        
        # Layer 4: Conv(3, 256), 3 x ResBlock(3, 256) / 5
        self.layer4 = ConvResBlock(in_channels=256, out_channels=256, kernel_size=3, downsample_factor=5)
        
        # Layer 5: Conv(3, 512), 3 x ResBlock(3, 512) / 5
        self.layer5 = ConvResBlock(in_channels=256, out_channels=512, kernel_size=3, downsample_factor=5)
        
        # Layer 6: Conv(3, 512), 3 x ResBlock(3, 512) / 5
        self.layer6 = ConvResBlock(in_channels=512, out_channels=512, kernel_size=3, downsample_factor=5)
        
        # Layer 7: Drop(0.5), FC(1024)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 1024)
        
        # Layer 8: Linear(6)
        self.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, sequence_length)
               Expected: (batch, 3, 1000)
        
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Layer 1: Conv(3, 128) / 2
        # (batch, 3, 1000) -> (batch, 128, 500)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.downsample1(x)
        
        # Layer 2: Conv(3, 128), 3 x ResBlock(3, 128) / 2
        # (batch, 128, 500) -> (batch, 128, 250)
        x = self.layer2(x)
        
        # Layer 3: Conv(3, 256), 3 x ResBlock(3, 256) / 2
        # (batch, 128, 250) -> (batch, 256, 125)
        x = self.layer3(x)
        
        # Layer 4: Conv(3, 256), 3 x ResBlock(3, 256) / 5
        # (batch, 256, 125) -> (batch, 256, 25)
        x = self.layer4(x)
        
        # Layer 5: Conv(3, 512), 3 x ResBlock(3, 512) / 5
        # (batch, 256, 25) -> (batch, 512, 5)
        x = self.layer5(x)
        
        # Layer 6: Conv(3, 512), 3 x ResBlock(3, 512) / 5
        # (batch, 512, 5) -> (batch, 512, 1)
        x = self.layer6(x)
        
        # Flatten: (batch, 512, 1) -> (batch, 512)
        x = x.squeeze(-1)
        
        # Layer 7: Drop(0.5), FC(1024)
        # (batch, 512) -> (batch, 1024)
        x = self.dropout(x)
        x = self.relu(self.fc(x))
        
        # Layer 8: Linear(6)
        # (batch, 1024) -> (batch, 6)
        x = self.classifier(x)
        
        return x


if __name__ == "__main__":
    # Test the model
    model = CNNModel(num_channels=3, num_classes=6)
    
    # Print model summary
    print("CNN Model Architecture (Capture24 Paper)")
    print("=" * 60)
    
    # Test with sample input
    batch_size = 4
    x = torch.randn(batch_size, 3, 1000)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass with intermediate shapes
    with torch.no_grad():
        out = model.relu(model.bn1(model.conv1(x)))
        out = model.downsample1(out)
        print(f"After Layer 1 (Conv /2): {out.shape}")
        
        out = model.layer2(out)
        print(f"After Layer 2 (ConvResBlock /2): {out.shape}")
        
        out = model.layer3(out)
        print(f"After Layer 3 (ConvResBlock /2): {out.shape}")
        
        out = model.layer4(out)
        print(f"After Layer 4 (ConvResBlock /5): {out.shape}")
        
        out = model.layer5(out)
        print(f"After Layer 5 (ConvResBlock /5): {out.shape}")
        
        out = model.layer6(out)
        print(f"After Layer 6 (ConvResBlock /5): {out.shape}")
        
        out = out.squeeze(-1)
        out = model.dropout(out)
        out = model.relu(model.fc(out))
        print(f"After FC(1024): {out.shape}")
        
        out = model.classifier(out)
        print(f"After Linear(6): {out.shape}")
    
    # Full forward pass
    output = model(x)
    print(f"\nFinal output shape: {output.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
