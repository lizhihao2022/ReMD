import functools

import torch
import torch.nn.functional as F

from utils.loss import LossRecord
from utils.metrics import get_obj_from_str

from .base import BaseTrainer


class ReMDTrainer(BaseTrainer):
    def build_model(self, **kwargs):
        self.remd_cfg = self.args["remd"]

        model_params = self.remd_cfg["model"]["params"]
        model = get_obj_from_str(self.remd_cfg["model"]["target"])(**model_params)

        diffusion_params = self.remd_cfg["diffusion"]["params"]
        self.base_diffusion = get_obj_from_str(self.remd_cfg["diffusion"]["target"])(**diffusion_params)
        model.remd_correction = self.base_diffusion.S
        return model

    def train(self, epoch, **kwargs):
        loss_record = LossRecord(["train_loss"])
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.model.train()

        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)
            x = F.interpolate(x, size=y.shape[2:], mode="bicubic", align_corners=False)

            micro_data = {
                "lq": x,
                "gt": y,
            }
            batch_size = x.shape[0]
            timesteps = torch.randint(
                0,
                self.base_diffusion.num_timesteps,
                size=(micro_data["gt"].shape[0],),
                device=x.device,
            )
            model_kwargs = {"lq": micro_data["lq"]}

            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data["gt"],
                micro_data["lq"],
                timesteps,
                first_stage_model=None,
                model_kwargs=model_kwargs,
                noise=None,
            )
            losses, _, _ = compute_losses()
            loss = losses["mse"]
            loss_record.update({"train_loss": loss}, n=batch_size)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        return loss_record

    def inference(self, x, y, **kwargs):
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=y.shape[2:], mode="bicubic", align_corners=False)

        model_kwargs = {"lq": x}
        y_pred = self.base_diffusion.p_sample_loop(
            y=x,
            model=self.model,
            first_stage_model=None,
            noise=None,
            clip_denoised=None,
            model_kwargs=model_kwargs,
            device=x.device,
            progress=True,
        )

        return y_pred.permute(0, 2, 3, 1)
