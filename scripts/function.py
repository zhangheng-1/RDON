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


    
# 定义 RealNVP 模块
class RealNVP(nn.Module):
    def __init__(self, layers, activation, kernel_initializer):
        super(RealNVP, self).__init__()
        # 从 layers 中提取输入维度 input_dim 和隐藏层维度 hidden_dims
        self.input_dim = layers[0]  # 输入维度
        self.split = self.input_dim // 2 
        layers[0] = self.split

        # 定义 MLP 分支，用于计算缩放因子 s 和平移因子 t
        self.mlp = MLP(layers, activation, kernel_initializer)

    def forward(self, inputs):
        ## 将输入分成 lower = [x1, x2, ..., x50] 和 upper = [x51, x52, ..., x100]
        lower = inputs[:, :self.split]  # 前半部分
        upper = inputs[:, self.split:]  # 后半部分
        # 通过 MLP 计算缩放因子 s 和平移因子 t
        s_t = self.mlp(lower)
        s, t = torch.split(s_t, self.split, dim=1) ## 输出沿着某个轴（通常是最后一个轴）分割成两个相等的部分
        # s = s_t[:, :self.split]  # 缩放因子
        # t = s_t[:, self.split:]  # 平移因子
        # 对后半部分 upper 进行变换：upper = upper * exp(s) + t，输出：[x51', x52', ..., x100']
        upper = upper * torch.exp(s) + t

        # 拼接前半部分和变换后的后半部分，输出：[x1, x2, ..., x50, x51', x52', ..., x100']
        outputs = torch.cat([lower, upper], dim=1)
        return outputs

    def inverse(self, inputs):
        lower = inputs[:, :self.split]  # 前半部分
        upper = inputs[:, self.split:]  # 后半部分
   
        # 通过 MLP 计算缩放因子 s 和平移因子 t
        s_t = self.mlp(lower)
        s = s_t[:, :self.split]  # 缩放因子
        t = s_t[:, self.split:]  # 平移因子
        # 对后半部分进行逆变换
        upper = (upper - t) * torch.exp(-s)
        # 拼接前半部分和逆变换后的后半部分
        outputs = torch.cat([lower, upper], dim=1)
        return outputs


# 定义 Reverse 模块
class Reverse(nn.Module):
    def __init__(self, input_dim):
        super(Reverse, self).__init__()
        # 创建一个反转的索引
        self.perm = torch.arange(input_dim)  # 正向索引
        self.perm = torch.flip(self.perm, dims=[0])  # 反转索引

    def forward(self, inputs):
        # 承接RealNVP1的输出为输入：[x1, x2, ..., x50, x51', x52', ..., x100']
        # 使用反转的索引对输入数据进行反转，输出：[x100', x99', ..., x52', x51', x50, x49, ..., x1]
        # 反转操作是自逆的
        return inputs[:, self.perm]
    
    def inverse(self, inputs):
        # 逆操作也是反转，因为反转是自逆的
        return inputs[:, self.perm]
    
class RandomShuffle(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 生成一次性的随机排列并固化
        perm = torch.randperm(input_dim)
        self.register_buffer('perm', perm)  # 注册为持久化缓冲区

    def forward(self, x):
        # 前向传播：按固定随机序打乱
        return x[:, self.perm]
    
    def inverse(self, x):
        # 逆向传播：执行完全相同的打乱操作
        return x[:, self.perm]  # 注意这不是数学可逆操作
    

# 定义 Serial 模块
class Serial(nn.Module):
    def __init__(self, *blocks):
        super(Serial, self).__init__()
        # 将所有模块存储在一个 ModuleList 中
        self.blocks = nn.ModuleList(blocks)

    def forward(self, inputs):
        # 前向计算：依次应用每个模块
        for block in self.blocks:
            inputs = block(inputs)
        return inputs

    def inverse(self, inputs):
        # 逆向计算：反向依次应用每个模块的逆操作
        for block in reversed(self.blocks):
            inputs = block.inverse(inputs)
        return inputs
    
"""
   由于 RealNVP 和 Reverse 模块都是可逆的，整个 Serial 模块也是可逆的。逆向计算时，网络会按照相反的顺序依次应用每个模块的逆操作。
   网络的功能:
   (1)前向计算：将输入数据通过一系列可逆变换映射到一个复杂的分布。
   (2)逆向计算：从复杂的分布中恢复原始输入数据。

   这种网络结构可以用于生成模型（如变分自编码器、生成对抗网络）或概率密度估计。

   通过Serial由 n 个 RealNVP 和 n 个 Reverse 模块交替堆叠而成。
   网络的核心功能是通过可逆变换将输入数据映射到一个复杂的分布，同时保持可逆性以便进行逆向计算。这种结构非常适合用于生成模型或概率密度估计任务。
"""





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
        input_dim,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        base_model = "MLP"    # or Modified_MLP 
    ):
        super().__init__()
        self.activation_trunk = self.activation_trunk = _get_act(activation)
        base_model= MLP if base_model=="MLP" else Modified_MLP

        realnvp = RealNVP(layer_sizes_branch,activation, kernel_initializer)
        reverse = Reverse(input_dim)
        self.branch = Serial(*([realnvp, reverse] * 5))

        self.trunk = base_model(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self, x, t, b):
        x_loc = torch.stack([x, t], dim=-1)
        x_func = b
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        x_loc = self.activation_trunk(self.trunk(x_loc))

        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        # x  = torch.sum(x_func*x_loc , dim=1)
        x = torch.einsum('ik, ilk -> il', x_func, x_loc)
        #x = np.dot(x_loc,x_func)  ## T的大小为（5000，200，100），B的大小为（5000，100）


        # Add bias
        x += self.b
        return x


