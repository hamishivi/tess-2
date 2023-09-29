import sys

import torch
from matplotlib import pyplot as plt

from sdlm.models.cdcd.cdf import LossCDF
from sdlm.schedulers.scheduling_simplex_ddpm import TokenWiseSimplexDDPMScheduler

module = LossCDF(100)

weights = torch.load(sys.argv[1], map_location="cpu")

l_t = weights["cdf.l_t"]
l_u = weights["cdf.l_u"]

module.load_state_dict({"l_t": l_t, "l_u": l_u})

scheduler = TokenWiseSimplexDDPMScheduler(
    num_train_timesteps=5000,
    beta_schedule="squaredcos_improved_ddpm",
    simplex_value=5,
    clip_sample=False,
    device=torch.device("cpu"),
)

with torch.no_grad():
    t = torch.linspace(0, 1, 100)
    u = module(t=t[None,], normalized=True)
    u = torch.clamp((u * 5000).long(), 0, 4999)
    plt.clf()
    # plt.plot(t.detach().cpu().numpy(), u.detach().cpu().numpy())
    # map to warped noise variance
    alphas_cumprod_timesteps = scheduler.alphas_cumprod[u]
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod_timesteps) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.squeeze()
    plt.plot(t.detach().cpu().numpy(), sqrt_one_minus_alpha_prod.detach().cpu().numpy())
    plt.savefig("loss_cdf.jpg")
