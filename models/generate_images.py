# Adapted from lumalabs/tvm under the license in https://github.com/lumalabs/tvm/blob/main/LICENSE
# Source: https://github.com/lumalabs/tvm/blob/main/generate_images.py
# Adjusted for this project needs.

import inspect

import torch

 
 

# ----------------------------------------------------------------------------
 
 
 
 
@torch.no_grad()
def generator_fn(net, latents, class_labels=None, num_steps=None, cfg_scale=None, lq_latent_cond=None, use_time_gap=True):
    # Time step discretization. 
    num_steps = 1 if num_steps is None else num_steps
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64, device=latents.device) 

 
    # Sampling steps
    x =  latents.to(torch.float64)   


    beta_scale = (1 / torch.as_tensor(cfg_scale, device=x.device, dtype=torch.float64).view(-1, 1, 1,
                                                                                            1)) if cfg_scale is not None else None
    beta_scale = beta_scale.repeat(x.shape[0], 1, 1, 1) if beta_scale is not None else torch.ones_like(t_steps)[:1]

    out_scale = 1.0
    if cfg_scale is not None and getattr(net, "do_scale_param", False):
        out_scale = cfg_scale

    forward_params = inspect.signature(net.forward_model).parameters
    supports_second_time = "s" in forward_params

        
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):

        # torch.cuda.empty_cache()
        if supports_second_time:
            forward_kwargs = dict(
                class_labels=class_labels,
                beta_scale=beta_scale,
                lq_latent_cond=lq_latent_cond,
                use_time_gap=use_time_gap,
            )
            F_ts = net.forward_model(x, t_cur, t_next, **forward_kwargs).to(torch.float64) * out_scale
        else:
            F_ts = net.forward_model(x, t_cur).to(torch.float64) * out_scale
        x = x + (t_next - t_cur) * F_ts  
         
    return x


@torch.no_grad()
def generator_fn_fm_only(
        net,
        latents,
        class_labels=None,
        num_steps=None,
        cfg_scale=None,
        lq_latent_cond=None,
        use_time_gap=False,
):
    # Time step discretization.
    num_steps = 1 if num_steps is None else num_steps
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64, device=latents.device)
    dt = t_steps[1] - t_steps[0]


    # Sampling steps
    x = latents.to(torch.float64)
    beta_scale = (1 / torch.as_tensor(cfg_scale, device=x.device, dtype=torch.float64).view(-1, 1, 1,
                                                                                            1)) if cfg_scale is not None else None
    beta_scale = beta_scale.repeat(x.shape[0], 1, 1, 1) if beta_scale is not None else torch.ones_like(t_steps)[:1]

    out_scale = 1.0
    if cfg_scale is not None and net.do_scale_param:
        out_scale = cfg_scale
    forward_params = inspect.signature(net.forward_model).parameters

    for i, t_cur in enumerate(t_steps[:-1]):
        forward_kwargs = dict(
            class_labels=class_labels,
            beta_scale=beta_scale,
            use_time_gap=False,
            lq_latent_cond=lq_latent_cond,
        )
        target_s = t_cur
        F_ts = net.forward_model(x, t_cur, target_s, **forward_kwargs).to(
            torch.float64
        ) * out_scale
        x = x + dt * F_ts

    return x


def normalize_validation_sampler(mode):
    mode = str(mode or "fm_only").lower().replace("-", "_")
    aliases = {
        "fm": "fm_only",
        "diag": "fm_only",
        "diagonal": "fm_only",
        "fm_diag": "fm_only",
        "fm_only": "fm_only",
        "jump": "jump",
        "tvm": "jump",
        "offdiag": "jump",
        "off_diagonal": "jump",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported validation sampler {mode!r}; expected 'fm_only' or 'jump'.")
    return aliases[mode]


def select_generator_fn(mode=None):
    return generator_fn if normalize_validation_sampler(mode) == "jump" else generator_fn_fm_only
