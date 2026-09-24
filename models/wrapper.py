# Adapted from lumalabs/tvm under the license in https://github.com/lumalabs/tvm/blob/main/LICENSE
# Source: https://github.com/lumalabs/tvm/blob/main/training/wrapper.py
# Adjusted for this project needs.

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from utils import persistence
from models.HDiT.create_arch import create_arch
from models.HDiT.hourglass import flags
from models.HDiT.hourglass.hdit_jvp import create_jvp_arch


@persistence.serializable_class
class GLOWFMWrapper(torch.nn.Module):

    def __init__(
            self,
            img_resolution,  # Image resolution.
            img_channels,  # Number of color channels.
            label_dim=0,  # Number of class labels, 0 = unconditional.
            mixed_precision=None,
            model_type="hdit16_XL2lat",
            data_type='bf16',
            detach_jvp=False,
            do_scale_param=False,
            use_checkpointing=False,
            **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()

        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.label_dim = label_dim
        if data_type == 'fp16':
            self.dtype = torch.float16
        elif data_type == 'bf16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.model_kwargs = model_kwargs
        self.use_checkpointing = bool(use_checkpointing)
        self.model = create_arch(model_type)

        print('Model # Mparams:', sum(p.numel() for p in self.model.parameters()) / 1000000)

    def forward_model(
            self,
            x,
            t,
            force_fp32=False,
    ):

        with torch.amp.autocast('cuda', enabled=(self.dtype != torch.float32) and not force_fp32, dtype=self.dtype):
            with flags.checkpointing(self.use_checkpointing):
                F_x = self.model(
                    x.to(torch.float32),
                    t.to(torch.float32).flatten(),
                )
        return F_x.to(t.dtype)


    def forward(
            self,
            xt,
            t,
            v_target,
            return_aux=False,
    ):

        F_ss = self.forward_model(
            xt.to(torch.float32),
            t.to(torch.float32).reshape(-1, 1, 1, 1))

        fm_loss = ((F_ss - v_target) ** 2).flatten(1).mean(-1)
        loss = fm_loss

        logs = {'ts': t.flatten(),
                'fm_loss': fm_loss.detach(),
                }
        if return_aux:
            logs["_pred_velocity"] = F_ss

        return loss, logs


@persistence.serializable_class
class TVMWHDiTWrapperJVP(torch.nn.Module):

    def __init__(
            self,
            img_resolution,  # Image resolution.
            img_channels,  # Number of color channels.
            label_dim=0,  # Number of class labels, 0 = unconditional.
            mixed_precision=None,
            model_type="hdit-jvp_XL2lat16",
            data_type='bf16',
            detach_jvp=False,
            do_scale_param=False,
            use_sdpa_jvp=False,
            use_checkpointing=False,
            **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()

        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.label_dim = label_dim
        if data_type == 'fp16':
            self.dtype = torch.float16
        elif data_type == 'bf16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.model_kwargs = model_kwargs
        self.model = create_jvp_arch(model_type)

        print('Model # Mparams:', sum(p.numel() for p in self.model.parameters()) / 1000000)

        self.do_scale_param = do_scale_param
        self.detach_jvp = detach_jvp
        self.use_sdpa_jvp = bool(use_sdpa_jvp)
        self.use_checkpointing = bool(use_checkpointing)

    def _second_time_cond(self, t, s):
        return s

    def forward_model(
            self,
            x,
            t,
            s,
            class_labels=None,
            beta_scale=None,
            force_fp32=False,
            use_time_gap=False,
            lq_latent_cond=None
    ):

        with torch.amp.autocast('cuda', enabled=(self.dtype != torch.float32) and not force_fp32, dtype=self.dtype):
            second_time_cond = self._second_time_cond(t, s)
            with flags.checkpointing(self.use_checkpointing):
                F_x = self.model(
                    x.to(torch.float32),
                    t.to(torch.float32).flatten(),
                    second_time_cond.to(torch.float32).flatten(),
                    use_sdpa_jvp=self.use_sdpa_jvp,
                )
        return F_x.to(t.dtype)

    def get_dfds(
            self,
            x,
            t,
            s,
            class_labels=None,
            beta_scale=None,
            force_fp32=False,
            **model_kwargs,
    ):
        def model_wrapper(x_, t_, s_):
            second_time_cond = self._second_time_cond(t_, s_)
            return self.model(x_, t_, second_time_cond, class_labels, beta_scale.flatten(), use_sdpa_jvp=True,
                              **model_kwargs)

        with torch.amp.autocast('cuda', enabled=(self.dtype != torch.float32) and not force_fp32, dtype=self.dtype):
            # JVP is taken only along the s input. That gives us dF/ds without forming a backward graph
            # through the teacher-style target construction used by the training objective.
            F_ts, F_ts_grad = torch.func.jvp(
                model_wrapper,
                (x, t.flatten(), s.flatten()),
                (torch.zeros_like(x), torch.zeros_like(s).flatten(), torch.ones_like(s).flatten())
            )
        F_ts = F_ts.to(t.dtype)
        F_ts_grad = F_ts_grad.to(t.dtype)
        if self.detach_jvp:
            F_ts_grad = F_ts_grad.detach()

        out_scale = 1.0
        if self.do_scale_param:
            w_scale = 1 / beta_scale.to(torch.float32)
            out_scale = w_scale
        # f_st is the explicit Euler-style update from time t to s, while df_ds is the tangent term
        # used by the jump-variance objective to match how that update changes as s moves.
        f_st = x + (s - t) * F_ts * out_scale
        df_ds = out_scale * (F_ts + (s - t) * F_ts_grad)
        return f_st, df_ds

    def forward(
            self,
            xt,
            xs_prime,
            t,
            s,
            s_prime,
            class_labels,
            beta_scale,
            v_target,
            teacher_model=None,
            alpha_factor=1.0,
            use_time_gap=False,
            lq_latent_cond=None,
            return_aux=False,
            **model_kwargs,
    ):
        """
        note: safer to wrap jvp inside DDP forward since PyTorch doesn't have good JVP support
        """
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=xt.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )

        alpha_value = float(alpha_factor)
        run_jvp = alpha_value > 0.0
        alpha_tensor = xt.new_tensor(alpha_value)

        # Do TVM Calc only when alpha > 0
        if run_jvp:
            f_ts, df_ds = self.get_dfds(
                xt.to(torch.float32),
                t.to(torch.float32).reshape(-1, 1, 1, 1),
                s.to(torch.float32).reshape(-1, 1, 1, 1),
                class_labels,
                beta_scale.to(torch.float32).reshape(-1, 1, 1, 1),
            )

            with torch.no_grad():
                F_ss_from_fts, F_ss_uncond = teacher_model.forward_model(
                    torch.cat([f_ts, xs_prime], dim=0).to(torch.float32),
                    torch.cat([s, s_prime], dim=0).to(torch.float32).reshape(-1, 1, 1, 1),
                    torch.cat([s, s_prime], dim=0).to(torch.float32).reshape(-1, 1, 1, 1),
                    torch.cat([class_labels, torch.zeros_like(class_labels)],
                              dim=0) if class_labels is not None else None,
                    beta_scale=torch.cat([beta_scale, torch.ones_like(beta_scale)], dim=0).to(torch.float32).reshape(-1,
                                                                                                                     1,
                                                                                                                     1,
                                                                                                                     1),
                ).chunk(2, dim=0)
        else:
            df_ds = torch.zeros_like(xt)
            F_ss_from_fts = torch.zeros_like(xt)
            F_ss_uncond = torch.zeros_like(xt)

        fm_target_s = s_prime
        F_ss = self.forward_model(
            xs_prime.to(torch.float32),
            s_prime.to(torch.float32).reshape(-1, 1, 1, 1),
            fm_target_s.to(torch.float32).reshape(-1, 1, 1, 1),
            class_labels,
            beta_scale=beta_scale.to(torch.float32).reshape(-1, 1, 1, 1),
            use_time_gap=use_time_gap,
            lq_latent_cond=lq_latent_cond,
        )

        w_scale = 1 / beta_scale.to(torch.float32)
        if self.do_scale_param:
            F_ss_from_fts = w_scale * F_ss_from_fts
            F_ss = w_scale * F_ss

        terminal_loss = (((df_ds - F_ss_from_fts) * beta_scale) ** 2).flatten(1).mean(-1)

        scaled_v_target = w_scale * v_target
        v_target = torch.where(
            torch.isclose(w_scale, torch.ones_like(w_scale)),
            scaled_v_target,
            scaled_v_target + (1 - w_scale) * F_ss_uncond,
        )
        fm_loss = (((F_ss - v_target) * beta_scale) ** 2).flatten(1).mean(-1)

        same_time = torch.isclose(s.reshape(-1), t.reshape(-1))
        terminal_loss = torch.where(same_time, torch.zeros_like(terminal_loss), terminal_loss)

        if run_jvp:
            loss = alpha_tensor * terminal_loss + fm_loss
        else:
            loss = fm_loss
            terminal_loss = torch.zeros_like(fm_loss)

        logs = {'ts': t.flatten(),
                'terminal_loss': terminal_loss.detach(),
                'fm_loss': fm_loss.detach(),
                }
        if return_aux:
            logs["_pred_velocity"] = F_ss
        return loss, logs
