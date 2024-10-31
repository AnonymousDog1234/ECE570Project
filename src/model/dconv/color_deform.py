import torch
import torch.nn as nn


class ColorDeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False, color_deform=True):
        super(ColorDeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.kernel_elements = self.kernel_size * self.kernel_size

        # Main convolution layer
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # Downsampling the concatenated input and reference channels
        self.channel_down = nn.Conv2d(inc * 2, inc, kernel_size=1, stride=1)

        # Convolution for generating offsets
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)

        # Modulation for deformable convolution
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)

        # Optional color deformable convolution
        self.color_deform = color_deform
        if self.color_deform:
            self.c_conv = nn.Conv2d(inc, kernel_size * kernel_size * inc, kernel_size=3, padding=1, stride=stride)

    def forward(self, x, ref):
        torch.cuda.empty_cache()
        input_tensor = x 
        reference_tensor = ref
        # Ensure input and reference have the same shape
        assert input_tensor.shape == reference_tensor.shape, f"Input shape {input_tensor.shape} and reference shape {reference_tensor.shape} must match."

        # Fuse input and reference tensors along the channel dimension
        fused = torch.cat([input_tensor, reference_tensor], dim=1)
        fused = self.channel_down(fused)
        assert fused.shape == input_tensor.shape, f"Fused shape {fused.shape} and input shape {input_tensor.shape} do not match."

        # Generate offsets from the fused tensor
        offset_map = self.p_conv(fused)
        modulation_map = None
        if self.modulation:
            modulation_map = torch.sigmoid(self.m_conv(fused))
        color_offset_map = None
        if self.color_deform:
            color_offset_map = torch.tanh(self.c_conv(fused))

        # Apply deformable convolution
        output_tensor = self.apply_deform_conv(input_tensor, offset_map, modulation_map, color_offset_map)
        
        return output_tensor

    def apply_deform_conv(self, input_tensor, offset_map, modulation_map=None, color_offset_map=None):
        """Applies deformable convolution given input, offset, and optional modulation and color offset maps."""
        dtype = torch.float16

        if self.padding:
            input_tensor = self.zero_padding(input_tensor)
        
        # Get sampling locations based on offset map
        sampling_locations = self.get_sampling_locations(offset_map, dtype)

        # Perform bilinear interpolation
        position_offset = self.bilinear_interpolate(input_tensor, sampling_locations, input_tensor.size(), modulation_map)
        
        # Add color deformation if applicable
        if self.color_deform:
            color_offset = self.apply_color_deformation(color_offset_map, position_offset.size())
            input_offset = color_offset + position_offset

        # Apply modulation if enabled
        if modulation_map is not None:
            modulation_offset = self.apply_modulation_deformation(modulation_map, input_offset.size())
            input_offset *= modulation_offset

        # Reshape the offset tensor
        input_offset = self._reshape_input_offset(input_offset)

        # Apply final convolution
        return self.conv(input_offset)

    def get_sampling_locations(self, offset_map, dtype):
        """Calculates sampling locations for deformable convolution."""
        _, _, height, width = offset_map.size()
        
        # Generate mesh grid for sampling
        grid_x, grid_y = torch.meshgrid(
            torch.arange(1, height + 1, dtype=dtype, device=offset_map.device),
            torch.arange(1, width + 1, dtype=dtype, device=offset_map.device),
        )
        
        grid_x = grid_x.flatten().view(1, 1, height, width).repeat(1, self.kernel_elements, 1, 1)
        grid_y = grid_y.flatten().view(1, 1, height, width).repeat(1, self.kernel_elements, 1, 1)

        # Reshape offset map to match the grid size
        grid = torch.cat([grid_x, grid_y], 1)
        split_kernal = (self.kernel_size - 1) // 2
        center_x, center_y = torch.meshgrid(torch.arange(-split_kernal, split_kernal + 1,  dtype=dtype, device=offset_map.device), 
                                            torch.arange(-split_kernal, split_kernal + 1,  dtype=dtype, device=offset_map.device))
        center = torch.cat([torch.flatten(center_x), torch.flatten(center_y)], 0).view(1, 2 * self.kernel_elements, 1, 1)
       
        # Get the sampling coordinates by adding the offset to the grid
        sampling_locations = grid + offset_map + center

        return sampling_locations

    def bilinear_interpolate(self, input_tensor, sampling_locations, size, modulation_map=None):
        """Applies bilinear interpolation for deformable sampling."""
        
        q_lt, q_rb, q_lb, q_rt = self.get_sampling_neighbors(sampling_locations, size)

        # Compute bilinear interpolation weights
        g_lt, g_rb, g_lb, g_rt = self.get_bilinear_weights(sampling_locations, q_lt, q_rb, q_lb, q_rt, size)

        # Gather pixel values from neighbors
        x_q_lt = self.get_position_offset(input_tensor, q_lt)
        x_q_rb = self.get_position_offset(input_tensor, q_rb)
        x_q_lb = self.get_position_offset(input_tensor, q_lb)
        x_q_rt = self.get_position_offset(input_tensor, q_rt)

        # Perform the weighted sum
        position_offset = g_lt * x_q_lt + g_rb * x_q_rb + g_lb * x_q_lb + g_rt * x_q_rt
 
        return position_offset


    def apply_color_deformation(self, color_offset_map, target_size):
        """Applies color deformation."""
        batch_size, channels, height, width, elements = target_size
        color_offset_map = color_offset_map.view(batch_size, channels, elements, height, width)
        return color_offset_map.permute(0, 1, 3, 4, 2)

    def apply_modulation_deformation(self, modulation_offset_map, target_size):
        """Applies modulation deformation."""
        _, channels, _, _, _ = target_size
        modulation_offset_map = modulation_offset_map.permute(0, 2, 3, 1).unsqueeze(dim=1)  # (b, 1, h, w, ks*ks)
        modulation_offset_map = modulation_offset_map.expand(-1, channels, -1, -1, -1)  # (b, c, h, w, ks*ks)

        return modulation_offset_map

    def get_sampling_neighbors(self, sampling_locations, size):
        """Get the neighboring pixels for bilinear interpolation."""
        _, _, height, width = size
        sampling_locations = sampling_locations.contiguous().permute(0, 2, 3, 1)

        q_lt = sampling_locations.detach().floor()
        q_rb = q_lt + 1

        q_ltx = q_lt[..., :self.kernel_elements]
        q_lty = q_lt[..., self.kernel_elements:]
        q_rbx = q_rb[..., :self.kernel_elements]
        q_rby = q_rb[..., self.kernel_elements:]
        
        q_lt = torch.cat([torch.clamp(q_ltx, 0, height - 1),
                          torch.clamp(q_lty, 0, width - 1)],dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rbx, 0, height - 1),
                          torch.clamp(q_rby, 0, width - 1)],dim=-1).long()
        
        q_ltx = q_lt[..., :self.kernel_elements]
        q_lty = q_lt[..., self.kernel_elements:]
        q_rbx = q_rb[..., :self.kernel_elements]
        q_rby = q_rb[..., self.kernel_elements:]
        
        q_lb = torch.cat([q_ltx, q_rby], dim=-1).long()
        q_rt = torch.cat([q_rbx, q_lty], dim=-1).long()

        return q_lt, q_rb, q_lb, q_rt

    def get_bilinear_weights(self, sampling_locations, q_lt, q_rb, q_lb, q_rt, size):
        """Calculates bilinear interpolation weights."""
        _, _, height, width = size
        sampling_locations = sampling_locations.contiguous().permute(0, 2, 3, 1)
        q_ltx = q_lt[..., :self.kernel_elements]
        q_lty = q_lt[..., self.kernel_elements:]
        q_rbx = q_rb[..., :self.kernel_elements]
        q_rby = q_rb[..., self.kernel_elements:]
        q_lbx = q_lb[..., :self.kernel_elements]
        q_lby = q_lb[..., self.kernel_elements:]
        q_rtx = q_rt[..., :self.kernel_elements]
        q_rty = q_rt[..., self.kernel_elements:]
        sl_x = sampling_locations[..., :self.kernel_elements]
        sl_y = sampling_locations[..., self.kernel_elements:]

        sampling_locations = torch.cat([torch.clamp(sl_x, 0, height - 1),
                                        torch.clamp(sl_y, 0, width - 1)], dim=-1)
        
        sl_x = sampling_locations[..., :self.kernel_elements]
        sl_y = sampling_locations[..., self.kernel_elements:]

        g_lt = ((1 + (q_ltx.type_as(sampling_locations) - sl_x)) * (
            1 + (q_lty.type_as(sampling_locations) - sl_y)
        )).unsqueeze(dim=1)
        g_rb = ((1 - (q_rbx.type_as(sampling_locations) - sl_x)) * (
            1 - (q_rby.type_as(sampling_locations) - sl_y)
        )).unsqueeze(dim=1)
        g_lb = ((1 + (q_lbx.type_as(sampling_locations) - sl_x)) * (
            1 - (q_lby.type_as(sampling_locations) - sl_y)
        )).unsqueeze(dim=1)
        g_rt = ((1 - (q_rtx.type_as(sampling_locations) - sl_x)) * (
            1 + (q_rty.type_as(sampling_locations) - sl_y)
        )).unsqueeze(dim=1)

        return g_lt, g_rb, g_lb, g_rt

    def get_position_offset(self, input_tensor, q):
        """Gathers position offset from the input tensor."""
        
        batch, height, width, _ = q.shape
        _, channels, input_height, input_width = input_tensor.shape
        
        # Reshape input tensor x from (b, c, h, w) to (b, c, h * w)
        input_flattened = input_tensor.view(batch, channels, input_height * input_width)

        # Split q into x and y coordinates
        q_x = q[..., :self.kernel_elements]
        q_y = q[..., self.kernel_elements:]
        
        # Calculate index for gathering, converting 2D coordinates into 1D indices
        indices = (q_x * input_width + q_y).long()

        # Expand indices across channels and reshape for gathering
        indices_expanded = indices.unsqueeze(1).expand(batch, channels, -1, -1, -1)
        indices_flat = indices_expanded.reshape(batch, channels, -1)

        # Gather pixel values using calculated indices
        position_offset = torch.gather(input_flattened, dim=-1, index=indices_flat)
        
        # Reshape gathered tensor into (b, c, h, w, N)
        position_offset = position_offset.view(batch, channels, height, width, self.kernel_elements)

        return position_offset
    
    def _reshape_input_offset(self, input_offset):
        """Reshapes the offset tensor for final convolution."""
        batch_size, channels, height, width, _ = input_offset.size()
        input_offset = input_offset.view(batch_size, channels, height, width, self.kernel_size, self.kernel_size) # (b, c, h, w, ks, ks)
        input_offset = input_offset.permute(0, 1, 2, 4, 3, 5).contiguous()  # (b, c, h, ks, w, ks)

        return input_offset.view(batch_size, channels, height * self.kernel_size, width * self.kernel_size)
