import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossNonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super().__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # Simplify the creation of g_x, g_b, g_d layers using a helper method
        self.g_x = self._create_conv_upsample_layer()
        self.g_b = self._create_conv_upsample_layer()
        self.g_d = self._create_conv_upsample_layer()

        # Create weight layers using another helper method
        self.W_x, self.W_b, self.W_d, self.W_xb, self.W_xd = [
            self._create_weight_layer() for _ in range(5)
        ]

        # theta and phi layers in a similar manner
        self.theta_x, self.theta_b, self.theta_d = [
            self._create_conv_layer() for _ in range(3)
        ]
        self.phi_x, self.phi_b, self.phi_d = [
            self._create_conv_upsample_layer() for _ in range(3)
        ]

        # Use batch normalization
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.bn2 = nn.BatchNorm2d(self.in_channels)

        # Output convolution layer
        self.out_conv = nn.Conv2d(self.in_channels, self.in_channels, 1)

        # Use a helper function for these convolutions
        self.t = self._create_conv_no_bias()
        self.p = self._create_conv_no_bias()

    def _create_conv_upsample_layer(self):
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        )

    def _create_weight_layer(self):
        return nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1)

    def _create_conv_layer(self):
        return nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)

    def _create_conv_no_bias(self):
        return nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, bias=False)

    def _dot_kernel(self, x):
        t = self.t(x)
        p = self.p(x)

        b, c, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)

        att = torch.matmul(torch.relu(t), torch.relu(p))
        att = (att + att.permute(0, 2, 1)) / 2
        d = torch.sum(att, dim=2)
        d[d != 0] = torch.sqrt(1.0 / d[d != 0])
        att = att * d.unsqueeze(1) * d.unsqueeze(2)

        return att

    def forward(self, x, ob, od):
        B, C, H, W = x.size()

        # Simplify g_x, g_b, g_d operations
        g_x = self.g_x(x).flatten(2).transpose(1, 2)
        g_b = self.g_b(ob).flatten(2).transpose(1, 2)
        g_d = self.g_d(od).flatten(2).transpose(1, 2)

        # Calculate attention using the new kernel
        f_x = self._dot_kernel(x)
        f_b = self._dot_kernel(ob)
        f_d = self._dot_kernel(od)

        # Combine results for self and cross-attention
        x_self = torch.bmm(f_x, g_x).transpose(1, 2).view(B, self.inter_channels, H, W)
        ob_self = torch.bmm(f_b, g_b).transpose(1, 2).view(B, self.inter_channels, H, W)
        od_self = torch.bmm(f_d, g_d).transpose(1, 2).view(B, self.inter_channels, H, W)

        x_ob_cross = torch.bmm(f_b, g_x).transpose(1, 2).view(B, self.inter_channels, H, W)
        x_od_cross = torch.bmm(f_d, g_x).transpose(1, 2).view(B, self.inter_channels, H, W)

        # Apply the weight layers
        x_self = self.W_x(x_self)
        ob_self = self.W_b(ob_self)
        od_self = self.W_d(od_self)
        x_ob_cross = self.W_xb(x_ob_cross)
        x_od_cross = self.W_xd(x_od_cross)

        # Batch normalization and final output
        od_final = self.bn1(od_self + x_ob_cross)
        ob_final = self.bn2(ob_self + x_od_cross)

        output = self.out_conv(od_final + ob_final + x_self)

        return output + x
