import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# from https://github.com/igul222/plaid/blob/main/train_cdcd.py
# CDCD loss cdf. Warping is the same for all tokens in sequence.
class LossCDF(nn.Module):
    def __init__(self, n_bins):
        super().__init__()
        # our buckets! These are basically logits
        self.l_t = nn.Parameter(torch.zeros([n_bins]) - float(np.log(n_bins)))
        self.l_u = nn.Parameter(torch.zeros([n_bins]) - float(np.log(n_bins)))

    def forward(
        self, t=None, u=None, normalized=True, t_min=0, t_max=1, l_t=None, l_u=None
    ):
        """t.shape: [n, l]"""
        bsz = t.shape[0] if t is not None else u.shape[0]
        seq_len = t.shape[1] if t is not None else u.shape[1]
        if l_t is None:
            l_t = self.l_t
            l_t = l_t.expand(bsz, seq_len, -1)
        if l_u is None:
            l_u = self.l_u
            l_u = l_u.expand(bsz, seq_len, -1)
        # apply softmax over logits to get partition of unit interval
        w_t = F.softmax(l_t, dim=-1)
        # add a small constant to avoid numerical issues / minimum bin size
        w_t = w_t + 1e-3
        # renormalize to the unit range
        w_t = w_t / w_t.sum(-1)[:, :, None]
        # instead of softmax, we use exp for output logits (to fit to loss values)
        w_u = l_u.exp()
        # same as above, if we normalize we are effectively doing a softmax
        w_u = w_u + 1e-3
        if normalized:
            w_u = w_u / w_u.sum(-1)[:, :, None]
        # The first bucket edge is zero, then its the cumsum of the edge points
        # this means e_t[0:1], e_t[1:2] etc gives us the edges of the buckets
        e_t = torch.cat(
            [
                torch.zeros(list(w_t.shape[:-1]) + [1], device=w_t.device),
                w_t.cumsum(dim=-1),
            ],
            dim=-1,
        )
        e_u = torch.cat(
            [
                torch.zeros(list(w_t.shape[:-1]) + [1], device=w_u.device),
                w_u.cumsum(dim=-1),
            ],
            dim=-1,
        )
        # if we have t, we want to map to u (= cross-entropy values)
        if t is not None:
            # flatten out t to 1d
            original_shape = t.shape
            # renormalize t to the unit range
            t_prime = (t - t_min) / (t_max - t_min)
            # find the bucket t lies in
            t_idx = (e_t <= t_prime[:, :, None]).long().sum(dim=-1) - 1
            # clamp to be safe? Does this ever fire?
            t_idx = t_idx.clamp(min=0, max=w_t.shape[-1] - 1)
            # The actual warping operation: find what % through e_t we are,
            # and use that to interpolate between the edges of the e_u bucket.
            u = torch.gather(e_u, -1, t_idx[:, :, None]).squeeze(-1) + (
                torch.gather(e_u, -1, t_idx[:, :, None] + 1).squeeze(-1)
                - torch.gather(e_u, -1, t_idx[:, :, None]).squeeze(-1)
            ) * (
                (t_prime - torch.gather(e_t, -1, t_idx[:, :, None]).squeeze(-1))
                / (
                    torch.gather(e_t, -1, t_idx[:, :, None] + 1).squeeze(-1)
                    - torch.gather(e_t, -1, t_idx[:, :, None]).squeeze(-1)
                )
            )
            # return back to the og shape!
            return u.view(original_shape)
        elif u is not None:
            # in this case, we have some timesteps and want to map them to warped timesteps
            # that  (learnt-ly) correspond to a linear reduction in cross-entropy
            original_shape = u.shape
            # find bucket edges as above. Clamping still doesnt make sense?
            u_idx = (e_u <= u[:, :, None]).long().sum(dim=-1) - 1
            u_idx = u_idx.clamp(min=0, max=w_u.shape[-1] - 1)
            # again, linearly interpolate
            t_prime = torch.gather(e_t, -1, u_idx[:, :, None]).squeeze(-1) + (
                torch.gather(e_t, -1, u_idx[:, :, None] + 1).squeeze(-1)
                - torch.gather(e_t, -1, u_idx[:, :, None]).squeeze(-1)
            ) * (
                (u - torch.gather(e_u, -1, u_idx[:, :, None]).squeeze(-1))
                / (
                    torch.gather(e_u, -1, u_idx[:, :, None] + 1).squeeze(-1)
                    - torch.gather(e_u, -1, u_idx[:, :, None]).squeeze(-1)
                )
            )
            # since e_u may not be normalized, we need(?) to renormalize
            t = t_prime * (t_max - t_min) + t_min
            return t.view(original_shape)
