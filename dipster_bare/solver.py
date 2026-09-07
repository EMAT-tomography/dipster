
import os
import sys
import time
from collections.abc import Iterable
import numpy as np
import logging
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia import losses as L
# from ray import train, tune
from . import grad, nets, tomo, params, util
from .reporting import Report

from tqdm.notebook import trange, tqdm

class Solver():
    def __init__(self, ts = None, fromdict = False):
        self.setup=False
        if not  fromdict:
            self.params = params.Params(ts)


    def eval(self, ref=None, save_period=100):
        self.params.evaluate = True
        if not ref is None:
            self.ref_vol = ref
            self.params.eval_vol = True
        self.params.save_period = save_period
        self.results = Report(self.params)
        self.params.wandb_name = self.results.name
        return self

    def _setfromconfig(self, config, params):
        self.params = params
        self.params.depth = config['depth']
        self.params.lr  = config['lr']
        self.params.gamma = config['gamma']
        self.params.batch_size = config['batch_size']

        self.params.style_size = int(np.sqrt(self.params.proj_size/config['option']))
        self.params.up_factor = int(np.sqrt(self.params.proj_size*config['option']))
        self.params.hypertrain = True

    def _setnet(self, config):
        self.params.depth = config['depth']
        self.params.lr  = config['lr']
        self.params.gamma = config['gamma']
        self.params.style_size = config['style_size']
        self.params.up_factor = config['up_factor']

    def _setup(self, ts= None):
        #working on the Assumption CUDA is always available
        self.params.dev = torch.device(self.params.dev)
        self.params.set_env()
        self.params.log()

        self.grad = grad.CustomGradient(self.params.proj_size,self.params.batch_size,self.params.output_nch)
        self.loss_fn = L.GemanMcclureLoss(reduction="mean")

        self.net = nets.Net(self.params).to(self.params.dev)
        if ts is not None:
            self.manifold = nets.Manifold(ts.times[self.params.frames-1], self.params.proj_size, self.params.dev)
        else:
            self.manifold = nets.Manifold(self.params.frames, self.params.proj_size, self.params.dev)

        p = nets.get_params(self.net)
        self.mapnet = nets.MappingNet(self.params).to(self.params.dev)
        p += self.mapnet.parameters()

        # Optionally torch.compile the pure-PyTorch compute submodules. The projection
        # (tomo.fp/bp via custom autograd) stays eager - it isn't Dynamo-traceable.
        if getattr(self.params, 'compile_net', False):
            self.net = torch.compile(self.net, mode=self.params.compile_mode)
            self.mapnet = torch.compile(self.mapnet, mode=self.params.compile_mode)

        self.optimizer = torch.optim.Adam(p, lr=self.params.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.params.step_size, gamma=self.params.gamma)

        # Per-frame affine forward-model correction. Created here but not added to
        # the optimizer until the affine phase begins (see train()),
        # _affine_active gates the phase-2 loss path.
        self.affine = nets.AffineCorrection(self.params.frames).to(self.params.dev)
        self._affine_active = False

        self._mapnet_frozen = False

        self.step = 0
        self.setup = True

    def train(self, ts, show_summary=True, save_path='', train_frames=None, eval_callback=None):
        if self.setup == False:
            self._setup(ts)

        best_loss = 1e8
        best_psnr = 0
        best_ssim = 0

        # Initialised up-front so the end-of-training prints don't NameError when
        # no save-period evaluation runs (e.g. during hyperparameter tuning).
        best_loss_step = best_psnr_step = best_ssim_step = 0

        # Optional held-out training split: restrict frame sampling to these indices.
        if train_frames is not None:
            train_frames = torch.as_tensor(train_frames, dtype=torch.long, device=self.params.dev)

        # Path to save temp models
        previous_path = None
        previous_path_psnr = None
        previous_path_ssim = None

        if show_summary:
            print('Input shape:', (self.params.latent_dim, 1, 1))
            # summary(self.mapnet, input_size=(self.params.latent_dim, 1, 1))
            print(self.mapnet)
            # summary(self.net, input=(self.params.input_nch, self.params.style_size, self.params.style_size))
            print(self.net)

        if len(ts.data.shape) < 4:
            ts.data = ts.data[:,:,:, None]

        while self.step < self.params.max_steps:

            self.optimizer.zero_grad()

            # Freeze MappingNet once we reach mapnet_freeze_step: requires_grad=False stops its
            # gradients (Adam then skips it), so only the CNN (and the affine, if active) keep training.
            if (self.params.mapnet_freeze_step is not None
                and not self._mapnet_frozen
                and self.step >= self.params.mapnet_freeze_step):
                for p in self.mapnet.parameters():
                    p.requires_grad_(False)
                self._mapnet_frozen = True
                print(f'Step #{self.step: 8d} - MappingNet weights frozen')

            # Phase transition: switch the per-frame affine correction on once it is
            # due, adding its parameters as a new optimizer group (Adam state for
            # net/mapnet is preserved). affine_start_step=None => never => phase 1 only.
            if (self.params.affine_start_step is not None
                    and not self._affine_active
                    and self.step >= self.params.affine_start_step):
                self.optimizer.add_param_group(
                    {'params': self.affine.parameters(), 'lr': self.params.affine_lr})
                self._affine_active = True
                print(f'Step #{self.step: 8d} - affine correction enabled '
                      f'({self.params.frames} frames, lr={self.params.affine_lr})')

            if not self._affine_active:
                # ===== Phase 1: per-line reconstruction loss (original path) =====
                # Split for validation and pick batch
                if train_frames is None:
                    i_batched_frames = torch.randint(0, self.params.frames, (self.params.batch_size,)).to(self.params.dev)
                else:
                    i_batched_frames = train_frames[torch.randint(0, len(train_frames), (self.params.batch_size,))]

                # Run forward
                i_batched_depths = torch.randint(0, self.params.proj_size, (self.params.batch_size,)).to(self.params.dev)
                out_set = self.reconstruct_slices(ts.angles[i_batched_frames], i_batched_depths, ts.times[i_batched_frames])
                angles = ts.angles[i_batched_frames]

                data_1d_ref = torch.zeros([self.params.proj_size, self.params.batch_size,1, self.params.output_nch]).to(self.params.dev)
                for i in range(self.params.batch_size):
                    data_1d_ref[:,i,:,:] = ts.data[:,i_batched_depths[i], i_batched_frames[i],:][:,None,:]

                # Get projection
                self.grad.X = out_set
                data_1d_rec = self.grad(angles)
                
                # Get TV
                tv_array = torch.permute(out_set, [3,1,2,0])
                tv = L.total_variation(tv_array)

                # Backpropogate and update weights
                total_loss = self.loss_fn(data_1d_rec,data_1d_ref) + (tv*self.params.noise_regularizer)
                total_loss = total_loss.sum()
                total_loss.backward()
            else:
                # ===== Phase 2: full-frame projection loss with per-frame affine =====
                # The clean model projection of a whole frame is assembled in one
                # autograd call (batch = proj_size depth-slices), warped by that
                # frame's affine, and matched to the measured 2-D projection. The
                # affine absorbs residual acquisition artefacts so the volume stays
                # clean; identity_penalty keeps the corrections minimal.
                dev = self.params.dev
                proj_size = self.params.proj_size

                # Take all the projection data and pick params.affine_frames_per_step of them for this step
                # Do test: use batch size instead as the fixed number for  affine_frames_per_step
                # It is likely that only one single vol can be computed at each step, because the CNN computational
                # graph must be kept in memory until the backward pass happens, and this is ~ 9 Go per volume for a
                # 128x128 proj size. So in the current implementation, only one volume can realistically be rec at each step
                f_pool = torch.arange(self.params.frames, device=dev) if train_frames is None else train_frames
                sel = f_pool[torch.randint(0, len(f_pool), (max(1, int(self.params.affine_frames_per_step)),))]

                # Because we compute the affine transform on the full slice, the depths necessarily spans the whole
                # proj_size
                depths = torch.arange(proj_size, device=dev)

                total_loss = 0
                for f in sel:
                    # Pick the angle and time of f
                    ang = ts.angles[f].reshape(1).expand(proj_size)
                    tim = ts.times[f].reshape(1).expand(proj_size)
                    vol = self.reconstruct_slices(ang, depths, tim)            # full volume slices [proj,proj,proj,ch]
                    # Single-angle full-volume projection: gives the [det, depth] image
                    # directly, avoiding the proj_size^3 diagonal sinogram of self.grad.
                    clean = grad.single_angle_fp(vol, ts.angles[f]).reshape(1, 1, proj_size, proj_size)  # [1,1,det,depth]
                    warped = self.affine.warp(clean, f.reshape(1))
                    ref = ts.data[:, :, f, 0].reshape(1, 1, proj_size, proj_size)
                    
                    tv = L.total_variation(torch.permute(vol, [3, 1, 2, 0]))
                    total_loss = total_loss + self.loss_fn(warped, ref).sum() + (tv * self.params.noise_regularizer).sum()
                
                # Backpropagate within for loop, so we make sure all times are seen for affine optim
                total_loss = total_loss + self.params.affine_regularizer * self.affine.identity_penalty()
                total_loss.backward()

            if total_loss<best_loss:
                best_loss = total_loss
                best_loss_step = self.step

            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            # Tuning hook: fires every save_period regardless of wandb evaluation,
            # so Optuna can read an intermediate validation metric and prune.
            if eval_callback is not None and (self.step % self.params.save_period == 0):
                eval_callback(self.step)

            if self.params.evaluate==True:
                if self.step % self.params.save_period == 0 or self.step == self.params.max_steps:
                    # Log losses
                    self.results._update_values('losses',
                                                times = [self.step, time.time()-self.results.start], 
                                                losses = [self.step, total_loss])
                    
                    # Log the per-frame affine correction (phase 2 only). Added to the
                    # report's _log before _evaluate so it ships in the same wandb publish.
                    if self._affine_active:
                        self.results.log_affine(self.step, self.affine.decompose(self.params.proj_size))

                    # Log metrics
                    metrics = self._evaluate(ts)
                    print(f'Step #{self.step: 8d} - Loss = {total_loss: 10.5g} - PSNR = {metrics["PSNR"]: 8.3g} - SSIM = {metrics["SSIM"]: 10.5g}')
                    
                    # Update model
                    torch.save(self.state_dict(), os.path.join(save_path, self.params.wandb_name + '_' + str(self.step) + '.pkl'))
                    if not (previous_path is None):
                        os.remove(previous_path)
                    previous_path = os.path.join(save_path, self.params.wandb_name + '_' + str(self.step) + '.pkl')

                    # Update best measurements models
                    if metrics["PSNR"]>best_psnr:
                        best_psnr = metrics["PSNR"]
                        best_psnr_step = self.step

                        torch.save(self.state_dict(), 
                                   os.path.join(save_path, self.params.wandb_name + '_best_psnr_' + str(self.step) + '.pkl'))
                        if not (previous_path_psnr is None):
                            os.remove(previous_path_psnr)
                        previous_path_psnr = os.path.join(save_path, self.params.wandb_name + '_best_psnr_' + str(self.step) + '.pkl')


                    if metrics["SSIM"]>best_ssim:
                        best_ssim = metrics["SSIM"]
                        best_ssim_step = self.step

                        torch.save(self.state_dict(), 
                                   os.path.join(save_path, self.params.wandb_name + '_best_ssim_' + str(self.step) + '.pkl'))
                        if not (previous_path_ssim is None):
                            os.remove(previous_path_ssim)
                        previous_path_ssim = os.path.join(save_path, self.params.wandb_name + '_best_ssim_' + str(self.step) + '.pkl')

        # Final report
        print(f'Best loss: {best_loss} at step #{best_loss_step: 8d}')
        print(f'Best PSNR: {best_psnr} at step #{best_psnr_step: 8d}')
        print(f'Best SSIM: {best_ssim} at step #{best_ssim_step: 8d}')



    @torch.no_grad()
    def _evaluate(self, ts):

        # Get reconstruction
        rand = torch.randint(0, self.params.frames, (1,)).item()
        rec = np.zeros((self.params.proj_size, self.params.proj_size))
        rand = torch.randint(0, self.params.frames,(1,)).to(self.params.dev)
        for i in range(self.params.proj_size):
            val = tomo.fp(self.reconstruct_slices(ts.angles[rand], [i], ts.times[rand]),ts.angles[rand]).squeeze()
            rec[:,i] = util.torch_to_np(val)
        ref = ts.data[:,:,rand,0].squeeze()

        # Get metrics and log results on projections
        metrics = self.results.update(self.step, rec, ref, 'tiltseries')

        # Get metrics and log results on volumes
        if self.params.eval_vol == True:
            # reconstruct a slice of the volume at halfway through the centre and at time = 0
            rec = self.reconstruct_slices(ts.angles[0], self.params.proj_size//2,ts.times[0]).squeeze()
            rec = util.torch_to_np(rec).squeeze()

            # Get metrics on volume
            self.results.update(self.step,rec, self.ref_vol[:,self.params.proj_size//2,:].squeeze(), 'volume')

        # Log results with optuna if hypertraining
        if self.params.hypertrain == True:
            pass
        else:
            self.results.publish()

        return metrics

    def reconstruct_slices(self, angles, depths, times):
        #Note all arrays must be same lenghth
        batch_val = 1
        if isinstance(depths, Iterable):
            if len(depths) > 1:
                batch_val = len(depths)
        manifold_output = self.manifold.get_value(angles, depths, times)
        mapnet_output = self.mapnet(manifold_output).reshape((batch_val,self.params.input_nch, self.params.style_size,self.params.style_size))
        out = self.net(mapnet_output)
        out= torch.permute(out, (2,0,3,1))
        return out


    def reconstruct(self, times=None, angles=None, **kwargs):
        #define kwargs ts is a tiltseries object
        # or you can just input the angles and times

        usesetangle = kwargs.get('usesetangle', False)
        ts = kwargs.get('ts', None)

        if times is None and angles is None:
            angles =  kwargs.get('angles', ts.angles)
            times = kwargs.get('times', ts.times)

        if usesetangle:
            angles = torch.zeros_like(times)

        depths = kwargs.get('depths', range(self.params.proj_size))
        save_func = kwargs.get('save_func', None)
        saveas = kwargs.get('saveas', None)

        with tqdm(range(len(times)), desc='Reconstructing') as pbar:
            for i in pbar:
                rec = np.zeros((self.params.proj_size, self.params.proj_size, len(depths)))
                tiled_angles = torch.full((len(depths),), angles[i]).to(self.params.dev)
                tiled_times = torch.full((len(depths),), times[i]).to(self.params.dev)
                rec[:,:,:] = util.torch_to_np(self.reconstruct_slices(tiled_angles, depths, tiled_times)).squeeze()

                pbar.set_postfix_str(f'{tiled_angles.shape} {tiled_times.shape} {len(depths)}')
                if (save_func is not None) and (saveas is not None):
                    save_func(rec, i, saveas)
                else:
                    yield rec, util.torch_to_np(times[i])

    def get_affine(self):
        """Learned per-frame 2x3 affine corrections as a CPU tensor [frames, 2, 3].

        Identity rows ``[[1,0,0],[0,1,0]]`` mean no correction; deviations read out
        the residual acquisition artefact (shift/scale/skew) absorbed for each frame.
        """
        return self.affine.theta.detach().cpu()

    def state_dict(self):
        try:
            return {'scheduler': self.scheduler.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    # Save the underlying module so keys stay un-prefixed whether or
                    # not the module was torch.compile'd (OptimizedModule adds _orig_mod.).
                    'net': getattr(self.net, '_orig_mod', self.net).state_dict(),
                    'mapnet': getattr(self.mapnet, '_orig_mod', self.mapnet).state_dict(),
                    'manifold': self.manifold.to_dict(),
                    'affine': self.affine.state_dict(),
                    'affine_active': self._affine_active,
                    'params':   self.params.to_dict(),
                    'step': self.step
                    # Add any other state you want to save
                    }
        except:
            return 0

    @classmethod
    def from_state_dict(cls, state_dict):
        solver = cls(fromdict=True)
        solver.params = params.Params.from_dict(state_dict['params'])
        solver._setup()

        # Restore the affine state. If it was active when saved, the optimizer had an
        # extra param group, so re-add it before loading optimizer state
        # Old checkpoints lack these keys and stay clean.
        if 'affine' in state_dict:
            solver.affine.load_state_dict(state_dict['affine'])
        solver._affine_active = state_dict.get('affine_active', False)
        if solver._affine_active:
            solver.optimizer.add_param_group(
                {'params': solver.affine.parameters(), 'lr': solver.params.affine_lr})

        solver.scheduler.load_state_dict(state_dict['scheduler'])
        solver.optimizer.load_state_dict(state_dict['optimizer'])
        
        # Load into the underlying module (matches the un-prefixed keys saved above),
        # so compiled and uncompiled checkpoints are interchangeable.
        getattr(solver.net, '_orig_mod', solver.net).load_state_dict(state_dict['net'])
        getattr(solver.mapnet, '_orig_mod', solver.mapnet).load_state_dict(state_dict['mapnet'])
        solver.manifold.load_state_dict(state_dict['manifold'])

        solver.step = state_dict['step']
        solver.params.evaluate = False
        solver.params.eval_vol  = False
        solver.params.hypertrain = False

        return solver
