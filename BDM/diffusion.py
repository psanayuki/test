import torch.nn as nn
import torch as th
from step_sample import create_named_schedule_sampler
import numpy as np
import math
import torch
import torch.nn.functional as F


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar in the limit of num_diffusion_timesteps. Beta schedules may be added, but should not be removed or changed once they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(num_diffusion_timesteps,lambda t: 1-np.sqrt(t + 0.0001),  )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(num_diffusion_timesteps, lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,)
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64)
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and produces the cumulative product of (1-beta) up to that part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):  ## 2000
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.initializer_range = 0.02
        self.apply(self._init_weights)

    
    # recbole SASRec
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    #     self.init_weights()

    # def init_weights(self):
    #     nn.init.xavier_normal_(self.w_1.weight)
    #     nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        
        self.initializer_range = 0.02
        self.apply(self._init_weights)

    
    # recbole
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    #     self.init_weights()

    # def init_weights(self):
    #     nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1,1,1,corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class MultiKernelConvLayer(nn.Module):
    def __init__(self, hidden_size, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=k, padding=k//2) for k in kernel_sizes])
        self.projection = nn.Conv1d(in_channels=len(kernel_sizes) * hidden_size, out_channels=hidden_size, kernel_size=1)
        self.initializer_range = 0.02
        self.hidden_size = hidden_size
        self.apply(self._init_weights)

        # Move all layers to GPU if available
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_outputs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        return x

class BehaviorFeatureExtractor(nn.Module):
    def __init__(self, hidden_size, num_behaviors):
        super().__init__()
        self.extractors = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_behaviors + 1)])
        # MLP
        self.behavior_mlp = nn.Sequential(
            nn.Linear(num_behaviors * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.num_behaviors = num_behaviors

    def forward(self, split_seq):
        # behavior_features = [extractor(x) for extractor in self.extractors]
        # behavior_features = torch.cat(behavior_features, dim=-1)
        # behavior_features = self.behavior_mlp(behavior_features)
        behavior_features = []
        for b in range(1, self.num_behaviors + 1):
            behavior_features.append(self.extractors[b](split_seq[b]))  # BFC

        # BFC
        behavior_features = torch.cat(behavior_features, dim=-1)  # [batch_size, num_behaviors * seq_len_per_behavior, hidden_size]
        # behavior_features = behavior_features.view(batch_size, -1)  # [batch_size, num_behaviors * hidden_size]
        behavior_features = self.behavior_mlp(behavior_features)  # [batch_size, hidden_size]
        return behavior_features

class ParallelHybridBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout, num_behaviors):
        super(ParallelHybridBlock, self).__init__()
        # TE
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.conv_layer = MultiKernelConvLayer(hidden_size, kernel_sizes=[5, 7, 9])  # M-S CNN

        # BFC
        self.behavior_extractor = BehaviorFeatureExtractor(hidden_size, num_behaviors)

        # Gate
        self.gate = nn.Linear(2 * hidden_size, hidden_size)

        
        self.attention_sublayer = SublayerConnection(hidden_size, dropout)
        self.conv_sublayer = SublayerConnection(hidden_size, dropout)
        self.ffn = PositionwiseFeedForward(hidden_size, dropout)
        self.output_sublayer = SublayerConnection(hidden_size, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, split_seq):
        
        attn_out = self.attention_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        conv_out = self.conv_sublayer(x, self.conv_layer)

        # BFC
        behavior_features = self.behavior_extractor(split_seq)

        # gate
        fuse_BC = torch.cat([conv_out, behavior_features], dim=-1)
        fuse_gate_BC = torch.sigmoid(self.gate(fuse_BC))
        fused_BC = fuse_gate_BC * conv_out + (1 - fuse_gate_BC) * behavior_features

        fuse_input = torch.cat([attn_out, fused_BC], dim=-1)
        fuse_gate = torch.sigmoid(self.gate(fuse_input))
        fused = fuse_gate * attn_out + (1 - fuse_gate) * fused_BC

        # FFN
        output = self.output_sublayer(fused, self.ffn)
        return self.dropout(output)

class MultiScaleTransformer(nn.Module):
    def __init__(self, args):
        super(MultiScaleTransformer, self).__init__()
        self.hidden_size = args.hidden_size #  128
        self.heads_list = args.n_heads
        self.dropout = args.dropout
        self.n_blocks = args.n_layers

        self.transformer_blocks = nn.ModuleList(
            [ParallelHybridBlock(self.hidden_size, self.heads_list, self.dropout, args.behavior_num) for _ in range(self.n_blocks)])

    def forward(self, hidden, mask, split_seq):
        for transformer in self.transformer_blocks:
            hidden = transformer(hidden, mask, split_seq)
        return hidden


class Diffu_xstart(nn.Module):
    def __init__(self, hidden_size, args):
        super(Diffu_xstart, self).__init__()
        self.hidden_size = hidden_size
        self.linear_item = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_xt = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size)
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim), SiLU(), nn.Linear(time_embed_dim, self.hidden_size))
        self.fuse_linear = nn.Linear(self.hidden_size*3, self.hidden_size)
        self.att = MultiScaleTransformer(args)
        # self.mlp_model = nn.Linear(self.hidden_size, self.hidden_size)
        # self.gru_model = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # self.gru_model = nn.GRU(self.hidden_size, self.hidden_size, num_layers=args.num_blocks, batch_first=True)
        self.lambda_uncertainty = args.lambda_uncertainty
        self.dropout = nn.Dropout(args.dropout)
        self.norm_diffu_rep = LayerNorm(self.hidden_size)
        self.args = args

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, x_t, t, mask_seq, split_seq):
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        x_t = x_t + emb_t
        
        # lambda_uncertainty = th.normal(mean=th.full(rep_item.shape, 1.0), std=th.full(rep_item.shape, 1.0)).to(x_t.device)
        
        lambda_uncertainty = th.normal(mean=th.full(rep_item.shape, self.lambda_uncertainty), std=th.full(rep_item.shape, self.lambda_uncertainty)).to(x_t.device)  ## distribution
        # lambda_uncertainty = self.lambda_uncertainty  ### fixed

        for b in range(1, self.args.behavior_num + 1):
            split_seq[b] = split_seq[b] + lambda_uncertainty * x_t.unsqueeze(1)
        
        ####  Attention
        rep_diffu = self.att(rep_item + lambda_uncertainty * x_t.unsqueeze(1), mask_seq, split_seq)
        rep_diffu = self.norm_diffu_rep(self.dropout(rep_diffu))
        out = rep_diffu[:, -1, :]


        ## rep_diffu = self.att(rep_item, mask_seq)  ## do not use
        ## rep_diffu = self.dropout(self.norm_diffu_rep(rep_diffu))  ## do not use
        
        ####
        
        #### GRU
        # output, hn = self.gru_model(rep_item + lambda_uncertainty * x_t.unsqueeze(1))
        # output = self.norm_diffu_rep(self.dropout(output))
        # out = output[:,-1,:]
        ## # out = hn.squeeze(0)
        # rep_diffu = None
        ####
        
        ### MLP
        # output = self.mlp_model(rep_item + lambda_uncertainty * x_t.unsqueeze(1))
        # output = self.norm_diffu_rep(self.dropout(output))
        # out = output[:,-1,:]
        # rep_diffu = None
        ###
        
        # out = out + self.lambda_uncertainty * x_t
        
        return out, rep_diffu


class BDM(nn.Module):
    def __init__(self, args,):
        super(BDM, self).__init__()
        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])

        self.noise_schedule = args.noise_schedule
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
         # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])
       
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)  ## lossaware (schedule_sample)
        self.timestep_map = self.time_map()
        self.rescale_timesteps = args.rescale_timesteps
        self.original_num_steps = len(betas)

        if args.dataset == 'kaggle':
            # 1 click 2 comment 3 buy
            self.behavior_noise_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.3, 0.2], dtype=torch.float32))
        else:
            # noise weight 1 click 2 comment 3 fav 4 buy
            self.behavior_noise_weights = nn.Parameter(torch.tensor([1.0, 0.4, 0.2, 0.3, 0.1], dtype=torch.float32))

        self.args = args

        self.xstart_model = Diffu_xstart(self.hidden_size, args)

    def get_betas(self, noise_schedule, diffusion_steps):
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)  ## array, generate beta
        return betas
    

    def q_sample(self, x_start, t, tagbeh_emb, weight, behavior_sequence=None, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape
        if behavior_sequence is not None:
            # behavior_weights = self.behavior_noise_weights[behavior_sequence]  # (B, L)
            # weighted_noise = noise * behavior_weights.unsqueeze(-1)  
            weighted_noise = noise * weight.unsqueeze(-1)  
            x_t = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * weighted_noise  # weight noise
            )
        else:
            x_t = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise  ## reparameter trick
            )  ## genetrate x_t based on x_0 (x_start) with reparameter trick
        # x_t = (
        #     _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        #     + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        #     * noise  ## reparameter trick
        # )  ## genetrate x_t based on x_0 (x_start) with reparameter trick

        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return th.where(mask==0, x_start, x_t)  ## replace the output_target_seq embedding (x_0) as x_t

    def time_map(self):
        timestep_map = []
        for i in range(len(self.alphas_cumprod)):
            if i in self.use_timesteps:
                timestep_map.append(i)
        return timestep_map

    # def scale_t(self, ts):
    #     map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
    #     new_ts = map_tensor[ts]
    #     # print(new_ts)
    #     if self.rescale_timesteps:
    #         new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
    #     return new_ts

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  ## \mu_t
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq, split_seq):
        model_output, _ = self.xstart_model(rep_item, x_t, self._scale_timesteps(t), mask_seq, split_seq)
        
        x_0 = model_output  ##output predict
        # x_0 = self._predict_xstart_from_eps(x_t, t, model_output)  ## eps predict
        
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)  ## x_start: candidante item embedding, x_t: inputseq_embedding + outseq_noise, output x_(t-1) distribution
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq, tagbeh_emb, split_seq, behavior_sequence=None):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq, split_seq)
        noise = th.randn_like(noise_x_t)

        behavior_weights = self.behavior_noise_weights[behavior_sequence]  # (B, L)
        noise = noise * behavior_weights.unsqueeze(-1)  # noise weight

        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))  # no noise when t == 0
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick
        # sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise + self.args.tag_weight * tagbeh_emb
        return sample_xt

    def reverse_p_sample(self, item_rep, noise_x_t, mask_seq, tagbeh_emb, split_seq, behavior_sequence=None):
        device = next(self.xstart_model.parameters()).device
        indices = list(range(self.num_timesteps))[::-1]
        
        for i in indices: # from T to 0, reversion iteration  
            t = th.tensor([i] * item_rep.shape[0], device=device)
            with th.no_grad():
                noise_x_t = self.p_sample(item_rep, noise_x_t, t, mask_seq, tagbeh_emb, split_seq, behavior_sequence=behavior_sequence)
        return noise_x_t 

    def forward(self, item_rep, item_tag, mask_seq, tagbeh_emb, behavior_sp_seq, global_h, split_seq, behavior_sequence=None):        
        noise = th.randn_like(item_tag)
        t, weights = self.schedule_sampler.sample(item_rep.shape[0], item_tag.device) ## t is sampled from schedule_sampler
        
        loc_embs_multiview = torch.zeros_like(item_tag)  # (batch_size, seq_len, emb_dim)
        #  behavior noise
        for b in range(1, self.args.behavior_types + 1):
            weight = self.behavior_noise_weights[b]
            x_t = self.q_sample(behavior_sp_seq[b][:,-1], t, tagbeh_emb, weight, behavior_sequence=behavior_sequence, noise=noise)
            loc_embs_multiview = loc_embs_multiview + x_t
        # global noise
        global_weight = self.behavior_noise_weights[behavior_sequence]
        h_t = self.q_sample(global_h, t, tagbeh_emb, global_weight, behavior_sequence=behavior_sequence, noise=noise)
        
        x_t = loc_embs_multiview + h_t
        x_0, item_rep_out = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq, split_seq)  ##output predict
        return x_0, item_rep_out, weights, t

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Behavior_Specific(nn.Module):
    def __init__(self, config):
        super(Behavior_Specific, self).__init__()
        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.behavior_types = config['behavior_types']
        self.max_seq_length = config['max_seq_length']
        self.hidden_size = config['hidden_size']
        self.dropout_rate = config['dropout']
        self.embed_dropout = nn.Dropout(self.dropout_rate)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.item_embeddings = nn.Embedding(self.item_num+1, self.hidden_size)
        self.position_embeddings = nn.Embedding(self.max_seq_length, self.hidden_size)

    def split(self, input_embs, input_bt):
        split_seq = {}
        for b in range(1, self.behavior_types + 1):
            # b mask
            bs_mask = (input_bt == b).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
            
            # b_emb
            loc_embs_b = input_embs * bs_mask  # (batch_size, seq_len, emb_dim)
            
            # sub-seq 
            split_seq[b] = loc_embs_b
        
        return split_seq
    
    def forward(self, input_embs, input_bt):
        behavior_sp_seq = {}
        for b in range(1, self.behavior_types + 1):
            zero_like_emb = torch.zeros_like(input_embs)  # (batch_size, seq_len, emb_dim)

            bs_mask = (input_bt == b).float().unsqueeze(-1)  # (batch_size, seq_len, 1)

            loc_embs_b = input_embs * bs_mask  # (batch_size, seq_len, emb_dim)

            valid_indices = torch.nonzero(bs_mask.squeeze(-1), as_tuple=True)  # (batch_size, seq_len)

            if valid_indices[0].numel() == 0:
                behavior_sp_seq[b] = zero_like_emb
                continue
            
            
            valid_embs = loc_embs_b[valid_indices[0], valid_indices[1]]  # (num_valid, emb_dim)
            
            # padding
            num_valid = valid_embs.size(0)
            seq_len = zero_like_emb.size(1)
            
            if num_valid > seq_len:
                valid_embs = valid_embs[-seq_len:] 
                num_valid = seq_len
            
            # valid embedding->right
            zero_like_emb[:, -num_valid:, :] = valid_embs.unsqueeze(0).expand(zero_like_emb.size(0), -1, -1)
            
            # sub-seq
            behavior_sp_seq[b] = zero_like_emb
        
        return behavior_sp_seq