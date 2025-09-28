import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_act(act):
    if callable(act):
        return act

    if act == 'tanh':
        func = torch.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    elif act == 'none':
        func = Identity()
    else:
        raise ValueError(f'{act} is not supported')
    return func

def _get_initializer(initializer: str = "Glorot normal"):

    INITIALIZER_DICT = {
        "Glorot normal": torch.nn.init.xavier_normal_,
        "Glorot uniform": torch.nn.init.xavier_uniform_,
        "He normal": torch.nn.init.kaiming_normal_,
        "He uniform": torch.nn.init.kaiming_uniform_,
        "zeros": torch.nn.init.zeros_,
    }
    return INITIALIZER_DICT[initializer]


class ActNorm(nn.Module):
    """
    ActNorm-class for activation normalization as described in Section 1.3 in Glow by Kingma & Dhariwal
    Found inspiration for class implementation <a href="https://github.com/axium/Glow-Pytorch/blob/master/actnorm.py">here</a>.
    """

    def __init__(self, no_channels):
        super(ActNorm, self).__init__()
        shape = (1, no_channels, 1, 1)
        self.initialized = False
        self.log_std = torch.nn.Parameter(torch.zeros(shape).float())
        self.mean = torch.nn.Parameter(torch.zeros(shape).float())

    def initialize(self, x):
        with torch.no_grad():
            x_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            variance = ((x - x_mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            log_std = torch.log(torch.sqrt(variance))

            self.log_std.data.copy_(-log_std.data)
            self.mean.data.copy_(-x_mean.data)
            self.initialized = True

    def apply_bias(self, x, backward):
        """
        Subtracting bias if forward, addition if backward
        """

        direction = -1 if backward else 1
        return x + direction * self.mean

    def apply_scale(self, x, backward):
        """
        Applying scale
        """

        direction = -1 if backward else 1
        return x * torch.exp(direction * self.log_std)

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)

        x = self.apply_bias(x, False)
        x = self.apply_scale(x, False)
        return x, self.log_std

    def backward(self, z):
        x = self.apply_scale(z, True)
        x = self.apply_bias(x, True)
        return x


class WeightNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(WeightNormConv2d, self).__init__()
        ## nn.utils.weight_norm在未来可能被弃用，推荐使用from torch.nn.utils.parametrizations import weight_norm
        self.conv = nn.utils.weight_norm(
            
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding))

    def forward(self, x):
        return self.conv(x)


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Input x is (N, C, H, W)
        # Want output to be (N, 4*C, H/2, W/2)
        # with the squeezing operation described in realNVP paper
        N, C, H, W = x.shape
        x = x.view(N, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(N, 4 * C, H // 2, W // 2)

    def backward(self, x):
        return UnSqueeze()(x)


class UnSqueeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape  # C will be 4*C of the original, and H and W are H//2 and W//2
        x = x.view(N, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(N, C // 4, 2 * H, 2 * W)

    def backward(self, x):
        return Squeeze()(x)


class ResidualConv2d(nn.Module):
    """
    Residual Links between MaskedConv2d-layers
    As described in Figure 5 in "Pixel Recurrent Neural Networks" by Aaron van den Oord et. al.
    """

    def __init__(self, in_dim):
        super(ResidualConv2d, self).__init__()
        self.net = nn.Sequential(
            WeightNormConv2d(in_dim, in_dim, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            WeightNormConv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            WeightNormConv2d(in_dim, in_dim, kernel_size=1, padding=0),
            nn.LeakyReLU())

    def forward(self, x):
        return self.net(x) + x


class ResidualCNN(nn.Module):
    """
     Residual CNN-class using residual blocks from "Pixel Recurrent Neural Networks" by Aaron van den Oord et. al.
    """

    def __init__(self, in_channels, out_channels, conv_filters=128, residual_blocks=8):
        super().__init__()
        modules = [WeightNormConv2d(in_channels, conv_filters, kernel_size=3, padding=1), nn.LeakyReLU()]
        modules += [ResidualConv2d(conv_filters) for _ in range(residual_blocks)]
        modules += [WeightNormConv2d(conv_filters, out_channels, kernel_size=3, padding=1)]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class CheckboardAffineCouplingLayer(nn.Module):
    """
    Coupling layer for RealNVP-class with checkboard mask
    """

    def __init__(self, in_channels, conv_filters=128, residual_blocks=8, top_condition=True, input_shape=(32, 32)):
        super(CheckboardAffineCouplingLayer, self).__init__()
        self.register_buffer('mask', self.get_mask(input_shape, top_condition))
        self.cnn = ResidualCNN(in_channels, 2 * in_channels, conv_filters, residual_blocks)
        self.scale = nn.Parameter(torch.tensor([1.]),
                                  requires_grad=True)  # log_scale is is scale*x + scale_shift, i.e. affine transformation
        self.scale_shift = nn.Parameter(torch.tensor([0.]), requires_grad=True)

    def get_mask(self, input_shape, top_condition):
        """
        Get checkboard mask
        """
        H, W = input_shape
        mask = np.arange(H).reshape(-1, 1) + np.arange(W)
        mask = np.mod(top_condition + mask, 2)
        mask = mask.reshape(-1, 1, H, W)
        return torch.from_numpy(mask).float()

    def forward(self, x):
        x_ = x * self.mask
        s, t = torch.chunk(self.cnn(x_), 2, dim=1)
        log_scale = self.scale * torch.tanh(s) + self.scale_shift

        t = t * (1.0 - self.mask)  # Will be zero for the non-dependant
        log_scale = log_scale * (1.0 - self.mask)  # Will be zero for the non-dependant

        z = x * torch.exp(log_scale) + t
        return z, log_scale

    def backward(self, z):
        z_ = z * self.mask
        s, t = torch.chunk(self.cnn(z_), 2, dim=1)

        log_scale = self.scale * torch.tanh(s) + self.scale_shift

        t = t * (1.0 - self.mask)  # Will be zero for the non-dependant
        log_scale = log_scale * (1.0 - self.mask)  # Will be zero for the non-dependant

        x = (z - t) * torch.exp(-log_scale)
        return x


class ChannelAffineCouplingLayer(nn.Module):
    """
    Coupling layer for RealNVP-class with channel-wise masking
    """

    def __init__(self, in_channels, conv_filters=128, residual_blocks=8, top_condition=True, input_shape=(32, 32)):
        super(ChannelAffineCouplingLayer, self).__init__()
        self.top_condition = top_condition

        self.cnn = ResidualCNN(2 * in_channels, 4 * in_channels, conv_filters, residual_blocks)
        self.scale = nn.Parameter(torch.tensor([1.]),
                                  requires_grad=True)  # log_scale is is scale*x + scale_shift, i.e. affine transformation
        self.scale_shift = nn.Parameter(torch.tensor([0.]), requires_grad=True)

    def forward(self, x):
        N, C, H, W = x.shape
        first_channels, second_channels = x[:, :C // 2], x[:, C // 2:]

        if self.top_condition:
            s, t = torch.chunk(self.cnn(first_channels), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            z = torch.cat((first_channels, second_channels * torch.exp(log_scale) + t), dim=1)
            jacobian = torch.cat((torch.zeros_like(log_scale), log_scale), dim=1)  # We only condition on firs
        else:
            s, t = torch.chunk(self.cnn(second_channels), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            z = torch.cat((first_channels * torch.exp(log_scale) + t, second_channels), dim=1)
            jacobian = torch.cat((log_scale, torch.zeros_like(log_scale)),
                                 dim=1)  # We only condition on first 1/2 channels, so we get the identity matrix I of shape S

        return z, jacobian

    def backward(self, z):
        N, C, H, W = z.shape
        first_channels, second_channels = z[:, :C // 2], z[:, C // 2:]

        if self.top_condition:
            s, t = torch.chunk(self.cnn(first_channels), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            x = torch.cat((first_channels, (second_channels - t) * torch.exp(-log_scale)), dim=1)

        else:
            s, t = torch.chunk(self.cnn(second_channels), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            x = torch.cat(((first_channels - t) * torch.exp(-log_scale), second_channels), dim=1)

        return x
    
class invChannelAffineCouplingLayer(nn.Module):
    """
    Coupling layer for RealNVP-class with channel-wise masking
    """

    def __init__(self, in_channels, conv_filters=128, residual_blocks=8, top_condition=True, input_shape=(32, 32)):
        super(invChannelAffineCouplingLayer, self).__init__()
        self.top_condition = top_condition

        self.cnn = ResidualCNN(2 * in_channels, 4 * in_channels, conv_filters, residual_blocks)
        self.scale = nn.Parameter(torch.tensor([1.]),
                                  requires_grad=True)  # log_scale is is scale*x + scale_shift, i.e. affine transformation
        self.scale_shift = nn.Parameter(torch.tensor([0.]), requires_grad=True)

    def forward(self, x):
        N, C, H, W = x.shape
        first_channels, second_channels = x[:, C // 2:], x[:, :C // 2]

        if self.top_condition:
            s, t = torch.chunk(self.cnn(first_channels), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            z = torch.cat((first_channels, second_channels * torch.exp(log_scale) + t), dim=1)
            jacobian = torch.cat((torch.zeros_like(log_scale), log_scale), dim=1)  # We only condition on firs
        else:
            s, t = torch.chunk(self.cnn(second_channels), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            z = torch.cat((first_channels * torch.exp(log_scale) + t, second_channels), dim=1)
            jacobian = torch.cat((log_scale, torch.zeros_like(log_scale)),
                                 dim=1)  # We only condition on first 1/2 channels, so we get the identity matrix I of shape S

        return z, jacobian

    def backward(self, z):
        N, C, H, W = z.shape
        first_channels, second_channels = z[:, C // 2:], z[:, :C // 2]

        if self.top_condition:
            s, t = torch.chunk(self.cnn(first_channels), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            x = torch.cat((first_channels, (second_channels - t) * torch.exp(-log_scale)), dim=1)

        else:
            s, t = torch.chunk(self.cnn(second_channels), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            x = torch.cat(((first_channels - t) * torch.exp(-log_scale), second_channels), dim=1)

        return x



class RealNVP(nn.Module):
    """
    RealNVP implemented with coupling layers
    """

    def __init__(self, input_shape):
        super().__init__()
        C, H, W = input_shape
        modules =[]
        # modules = [[CheckboardAffineCouplingLayer(C, top_condition=i % 2 == 0),ActNorm(C)] for i in range(2)]

        modules.append([Squeeze()])
        modules += [[ChannelAffineCouplingLayer(C, top_condition=i % 2 == 0), ActNorm(4 * C)] for i in range(1)]
        modules += [[invChannelAffineCouplingLayer(C, top_condition=i % 2 == 0), ActNorm(4 * C)] for i in range(1)]
        modules += [[ChannelAffineCouplingLayer(C, top_condition=i % 2 == 0), ActNorm(4 * C)] for i in range(1)]
        modules += [[invChannelAffineCouplingLayer(C, top_condition=i % 2 == 0), ActNorm(4 * C)] for i in range(1)]
    
        modules.append([UnSqueeze()])

        # modules += [[CheckboardAffineCouplingLayer(C, top_condition=i % 2 == 0), ActNorm(C)] for i in range(1)]

        modules = [layer for layer_types in modules for layer in layer_types]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        z = x
        log_det_jacobian = torch.zeros_like(x)

        for layer in self.net:
            if isinstance(layer, (Squeeze, UnSqueeze)):
                z = layer(z)
                log_det_jacobian = layer(log_det_jacobian)  # Need to reshape jacobian as well
            else:
                z, new_log_det_jacobian = layer(z)
                log_det_jacobian += new_log_det_jacobian

        return z

    def backward(self, z):
        x = z
        for layer in list(self.net)[::-1]:
            x = layer.backward(x)

        return x
    



class MLP(nn.Module):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.activation = _get_act(activation)
        initializer = _get_initializer(kernel_initializer)
        initializer_zero = _get_initializer("zeros")

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=torch.float32
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x
    



class DeepONetCartesianProd(nn.Module):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """
    def __init__(
        self,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        base_model = "MLP"    # or Modified_MLP 
    ):
        super().__init__()
        self.activation_trunk = self.activation_trunk = _get_act(activation)
        base_model= MLP if base_model=="MLP" else Modified_MLP

        self.branch = RealNVP((1,32,32)) 

        self.trunk = base_model(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self, x, t, b):
        x_loc = torch.stack([x, t], dim=-1)
        x_func = b

        # Branch net to encode the input function
        x_func = self.branch(x_func)
        x_func = x_func.reshape(-1, 1024)
        # Trunk net to encode the domain of the output function
        # x_loc = self.trunk(x_loc)
        x_loc = self.activation_trunk(self.trunk(x_loc))



        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum('ik, ilk -> il', x_func, x_loc)



        # Add bias
        x += self.b
        return x
