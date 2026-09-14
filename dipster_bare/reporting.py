import os
import time

import numpy as np
import wandb
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from . import util

class Report():
    def __init__(self, params):
        # Set Up reporting
        self.hypertrain = params.hypertrain
        self.tables = {}

        if not params.hypertrain:
            os.environ["WANDB_DIR"] = params.wandb_local_dir
            run = wandb.init(project=params.wandb_project)
            self.name = run.name
            self._log = {}

        else:
            self.name = 'report'

        self.start = time.time()

    def quantify(self, rec, ref):
        rec = util.torch_to_np(rec)
        ref = util.torch_to_np(ref)

        psnr_val = psnr(rec, ref, data_range = ref.max()-ref.min())
        ssim_val = ssim(rec, ref, data_range = ref.max()-ref.min())

        return {'PSNR': psnr_val, 'SSIM': ssim_val}

    def update(self, step, rec, ref, title):

        # Calculate and log metrics
        metrics = self.quantify(rec, ref)
        self._update_values(title, PSNR=[step, metrics['PSNR']], SSIM=[step, metrics['SSIM']])

        # Log images
        rec = (rec - rec.min())/(rec.max()-rec.min())
        ref = (ref - ref.min())/(ref.max()-ref.min())

        self._update_images(title + '_tiltseries', [ref, rec])

        return metrics

    def _update_values(self, title, **kwargs):

        for key, value in kwargs.items():
            dict_key = title + "_" + key
            if dict_key not in self.tables:
                self.tables[dict_key] = [value]
            else:
                self.tables[dict_key].append(value)

            # Logs only if no optuna
            if not self.hypertrain:
                self._log[dict_key] = wandb.plot.line(wandb.Table(data=self.tables[dict_key], columns=["steps", dict_key]), "steps", dict_key, title=dict_key)

    def log_affine(self, step, components):
        """Log the per-frame affine correction to wandb (no-op during hypertraining).

        components is the dict from AffineCorrection.decompose() (per-frame [F]
        tensors). For each component this logs both a per-frame profile panel
        (value vs frame index) and aggregate mean / max-abs trend
        lines vs step.
        """
        if self.hypertrain:
            return
        n = len(next(iter(components.values())))
        frames = list(range(n))
        mean_kw, maxabs_kw = {}, {}
        for name, vals in components.items():
            vals = [float(v) for v in util.torch_to_np(vals)]
            table = wandb.Table(data=list(zip(frames, vals)), columns=['frame', name])
            self._log[f'affine_profile/{name}'] = wandb.plot.line(
                table, 'frame', name, title=f'affine {name} (step {step})')
            mean_kw[name] = [step, float(np.mean(vals))]
            maxabs_kw[name] = [step, float(np.max(np.abs(vals)))]
        self._update_values('affine_mean', **mean_kw)
        self._update_values('affine_maxabs', **maxabs_kw)

    def _update_images(self, title, value):
        if not self.hypertrain:
            self._log[title] = [wandb.Image(np.expand_dims(util.torch_to_np(value[0]), axis=-1), caption='reference'), wandb.Image(np.expand_dims(value[1], axis=-1), caption='reconstruction')]
        else:
            self.tables[title +'_image'] = value

    def publish(self):
        # print(self._log)
        wandb.log(self._log)
