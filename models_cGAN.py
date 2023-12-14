import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        activation = nn.PReLU() if upsample else nn.ReLU()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        return torch.add(self.residual_block(x), x)

class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding='same'),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
    def forward(self, x):
        return self.upsample(x)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, use_dropout=False):
        super().__init__()
        self.conv, self.act = self._build_block(in_channels, out_channels, downsample)
        self.norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5) if use_dropout else None
    
    def _build_block(self, in_channels, out_channels, downsample):
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            act = nn.LeakyReLU(0.2, True)
        else:
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            act = nn.ReLU(True)
        return conv, act

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x) if self.dropout else x

class ResidualUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, use_dropout=False):
        super().__init__()
        self.unetblock, self.shortcut_conv, self.act = self._build_block(in_channels, out_channels, downsample)
        self.norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5) if use_dropout else None

    def _build_block(self, in_channels, out_channels, downsample):
        if downsample:
            unetblock = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            )
            shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            act = nn.LeakyReLU(0.2)
        else:
            unetblock = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            )
            shortcut_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            act = nn.ReLU()
        return unetblock, shortcut_conv, act

    def forward(self, x):
        shortcut = self.shortcut_conv(x)
        x = self.unetblock(x)
        x = torch.add(x, shortcut)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x) if self.dropout else x


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU())
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(64, 64) for _ in range(8)])
        self.conv2 = nn.Sequential(nn.Conv2d(64, out_channels, 3, 1, 1), nn.Tanh())
    
    def forward(self, x):
        temp = x
        x = self.conv1(x)
        x = self.res_blocks(x)
        final = self.conv2(torch.add(x, temp))
        return final
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, n_filters=64):
        super().__init__()
        # Initial downsampling
        self.downsample_initial = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2, True)
        )
        # Downsampling layers
        self.down_layers = nn.Sequential(
            UNetBlock(n_filters, n_filters*2, downsample=True, use_dropout=False),
            UNetBlock(n_filters*2, n_filters*4, downsample=True, use_dropout=False),
            UNetBlock(n_filters*4, n_filters*8, downsample=True, use_dropout=False),
            UNetBlock(n_filters*8, n_filters*8, downsample=True, use_dropout=False),
            UNetBlock(n_filters*8, n_filters*8, downsample=True, use_dropout=False),
            UNetBlock(n_filters*8, n_filters*8, downsample=True, use_dropout=False)
        )
        # Innermost layer
        self.downsample_inner = nn.Sequential(
            nn.Conv2d(n_filters*8, n_filters*8, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2, True)
        )
        self.upsample_inner = UNetBlock(n_filters*8, n_filters*8, downsample=False, use_dropout=False)
        # Upsampling layers
        self.up_layers = nn.Sequential(
            UNetBlock(n_filters*8*2, n_filters*8, downsample=False, use_dropout=True),
            UNetBlock(n_filters*8*2, n_filters*8, downsample=False, use_dropout=True),
            UNetBlock(n_filters*8*2, n_filters*8, downsample=False, use_dropout=True),
            UNetBlock(n_filters*8*2, n_filters*4, downsample=False, use_dropout=False),
            UNetBlock(n_filters*4*2, n_filters*2, downsample=False, use_dropout=False),
            UNetBlock(n_filters*2*2, n_filters, downsample=False, use_dropout=False)
        )
        # Final upsampling
        self.upsample_final = nn.Sequential(
            nn.ConvTranspose2d(n_filters*2, out_channels, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.Tanh()
        )
    
    def forward(self, x):
        d_initial = self.downsample_initial(x)
        d = [d_initial]
        for layer in self.down_layers:
            d.append(layer(d[-1]))
        d_inner = self.downsample_inner(d[-1])
        u = self.upsample_inner(d_inner)
        for layer in self.up_layers:
            u = layer(torch.cat([u, d[-1]], dim=1))
            d.pop()
        u_final = self.upsample_final(torch.cat([u, d_initial], dim=1))
        return u_final

class ResidualUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, n_filters=64):
        super().__init__()

        self.downsample_initial = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2)
        )
        # Downsampling layers with residual connections
        self.down_layers = nn.Sequential(
            ResidualUNetBlock(n_filters, n_filters * 2, downsample=True, use_dropout=False),
            ResidualUNetBlock(n_filters * 2, n_filters * 4, downsample=True, use_dropout=False),
            ResidualUNetBlock(n_filters * 4, n_filters * 8, downsample=True, use_dropout=False),
            ResidualUNetBlock(n_filters * 8, n_filters * 8, downsample=True, use_dropout=False),
            ResidualUNetBlock(n_filters * 8, n_filters * 8, downsample=True, use_dropout=False),
            ResidualUNetBlock(n_filters * 8, n_filters * 8, downsample=True, use_dropout=False)
        )
        # Innermost layer
        self.downsample_inner = nn.Sequential(
            nn.Conv2d(n_filters * 8, n_filters * 8, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2)
        )
        self.upsample_inner = nn.Sequential(
            nn.ConvTranspose2d(n_filters * 8, n_filters * 8, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(n_filters * 8), 
            nn.ReLU()
        )
        # Upsampling layers with residual connections
        self.up_layers = nn.Sequential(
            ResidualUNetBlock(n_filters * 8 * 2, n_filters * 8, downsample=False, use_dropout=True),
            ResidualUNetBlock(n_filters * 8 * 2, n_filters * 8, downsample=False, use_dropout=True),
            ResidualUNetBlock(n_filters * 8 * 2, n_filters * 8, downsample=False, use_dropout=True),
            ResidualUNetBlock(n_filters * 8 * 2, n_filters * 4, downsample=False, use_dropout=False),
            ResidualUNetBlock(n_filters * 4 * 2, n_filters * 2, downsample=False, use_dropout=False),
            ResidualUNetBlock(n_filters * 2 * 2, n_filters, downsample=False, use_dropout=False)
        )
        # Final upsampling
        self.upsample_final = nn.Sequential(
            nn.ConvTranspose2d(n_filters * 2, out_channels, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.Tanh()
        )
    
    def forward(self, x):
        d_initial = self.downsample_initial(x)
        d = [d_initial]
        for layer in self.down_layers:
            d.append(layer(d[-1]))
        d_inner = self.downsample_inner(d[-1])
        u_inner = self.upsample_inner(d_inner)
        u = u_inner
        for layer in self.up_layers:
            u = layer(torch.cat([u, d[-len(self.up_layers) + self.up_layers.index(layer)]], dim=1))
        u_final = self.upsample_final(torch.cat([u, d_initial], dim=1))
        return u_final
    
class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class PatchGAN(nn.Module):

    def __init__(self, in_channels, n_filters=[64, 128, 256, 512]):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, n_filters[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True))
        self.layer2 = ConvolutionBlock(n_filters[0], n_filters[1], stride=2)
        self.layer3 = ConvolutionBlock(n_filters[1], n_filters[2], stride=2)
        self.layer4 = ConvolutionBlock(n_filters[2], n_filters[3], stride=1)    
        self.layer5 = nn.Conv2d(n_filters[3], 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]:
            x = layer(x)
        return x

# Test the discriminator model (PatchGAN-Classifier)
if __name__ == "__main__":
    
    sample_input = torch.randn((1, 3, 256, 256))
    patchGAN_model = PatchGAN(in_channels=3)
    output = patchGAN_model(sample_input)
    
    print("PatchGAN Model Architecture:")
    print(patchGAN_model)
    print("\nOutput Shape:")
    print(output.shape)

    noise_vector = torch.randn((1, 1, 256, 256))
    model = UNet(in_channels=1, out_channels=2, n_filters=64)
    output = model(noise_vector)
    output_np = output.cpu().detach().numpy()
    combined_output = np.concatenate([noise_vector.cpu().detach().numpy(), output_np], axis=1)
    combined_output_transposed = np.transpose(combined_output[0], (1, 2, 0))

    plt.imshow(combined_output_transposed, cmap='gray')  
    plt.show()
