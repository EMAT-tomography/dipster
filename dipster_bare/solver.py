'''
Training methods for the optimisation of DIP-STER.
'''
import os
import sys
import time
import logging
from collections.abc import Iterable

import numpy as np
from tqdm.notebook import tqdm, trange

import torch
from  torch import nn
import torch.nn.functional as F

from kornia import losses as L

from . import grad, nets, tomo, params, util
from .reporting import Report


class Solver():
    def __init__(self, ts=None, fromdict=False):
        self.setup = False
        if not fromdict:
            self.params = params.Params(ts)

    def eval(self, ref=None, save_period=100):
        self.params.evaluate = True
        if ref is not None:
            self.ref_vol = ref
            self.params.eval_vol = True
        self.params.save_period = save_period
        self.results = Report(self.params)
        self.params.wandb_name = self.results.name
        return self

    def _setfromconfig(self, config, list_params):
        self.params = list_params
        self.params.depth = config['depth']
        self.params.lr = config['lr']
        self.params.gamma = config['gamma']
        self.params.batch_size = config['batch_size']

        self.params.style_size = int(np.sqrt(self.params.proj_size/config['option']))
        self.params.up_factor = int(np.sqrt(self.params.proj_size*config['option']))
        self.params.hypertrain = True

    def _setnet(self, config):
        self.params.depth = config['depth']
        self.params.lr = config['lr']
        self.params.gamma = config['gamma']
        self.params.style_size = config['style_size']
        self.params.up_factor = config['up_factor']

    def _setup(self, ts=None):
        # working on the Assumption CUDA is always available
        self.params.dev = torch.device(self.params.dev)
        self.params.set_env()
        self.params.log()

        self.grad = grad.CustomGradient(self.params.proj_size, self.params.batch_size, self.params.output_nch)
        self.loss_fn = L.GemanMcclureLoss(reduction="mean")

        self.net = nets.Net(self.params).to(self.params.dev)
        if ts is not None:
            self.manifold = nets.Manifold(ts.times[self.params.frames-1], self.params.proj_size, self.params.latent_dim, self.params.dev)
        else:
            self.manifold = nets.Manifold(self.params.frames, self.params.proj_size, self.params.latent_dim, self.params.dev)

        p = nets.get_params(self.net)
        self.mapnet = nets.MappingNet(self.params).to(self.params.dev)
        p += self.mapnet.parameters()
        self._mapnet_frozen = False

        # Optionally torch.compile the pure-PyTorch compute submodules. The projection
        # (tomo.fp/bp via custom autograd) stays eager - it isn't Dynamo-traceable.
        if getattr(self.params, 'compile_net', False):
            self.net = torch.compile(self.net, mode=self.params.compile_mode)
            self.mapnet = torch.compile(self.mapnet, mode=self.params.compile_mode)

        self.optimizer = torch.optim.Adam(p, lr=self.params.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.params.step_size, gamma=self.params.gamma) # Old method        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.params.gamma,
                                                                    patience=self.params.lr_patience, threshold=self.params.lr_threshold,
                                                                    cooldown=self.params.lr_cooldown, min_lr=self.params.lr_min_lr)

        # Per-frame affine forward-model correction. Created here but not added to
        # the optimizer until the affine phase begins (see train()),
        # _affine_active gates the phase-2 loss path.
        self.affine = nets.AffineCorrection(self.params.frames).to(self.params.dev)
        self._affine_active = False

        self.step = 0
        self.setup = True

    def train(self, ts, show_summary=False, save_path='', train_frames=None, eval_callback=None):
        if self.setup is False:
            self._setup(ts)

        # # Initialised up-front so the end-of-training prints don't NameError when
        # # no save-period evaluation runs (e.g. during hyperparameter tuning).
        best_trainloss = [-1, 1e8]
        best_intvalloss = [-1, 1e8]
        best_intvalssim = [-1, 0]
        best_intvalpsnr = [-1, 0]
        best_fullvalloss = [-1, 1e8]
        best_fullvalssim = [-1, 0]
        best_fullvalpsnr = [-1, 0]

        torch.manual_seed(self.params.seed)

        ###  - - Definition of the training and validation set - -
        assert 0.0 < self.params.training_split <= 1.0, "self.params.training_split must be between 0 and 1"
        all_frames = torch.arange(self.params.frames, device=self.params.dev)
        shuffled_indices = torch.randperm(len(all_frames), device=self.params.dev)

        n_train = int(len(shuffled_indices) * self.params.training_split)
        # n_val = (self.params.frames - n_train)
        self.train_indices = shuffled_indices[:n_train]
        self.val_indices = shuffled_indices[n_train:]

        logging.debug('train size: %s', n_train)
        logging.debug('train index: %s', self.train_indices)
        logging.debug('val index: %s', self.val_indices)

        n_train_batch = (n_train * self.params.proj_size + self.params.batch_size - 1) // self.params.batch_size

        if show_summary:
            print('Input shape:', (self.params.latent_dim, 1, 1))
            # summary(self.mapnet, input_size=(self.params.latent_dim, 1, 1))
            print(self.mapnet)
            # summary(self.net, input=(self.params.input_nch, self.params.style_size, self.params.style_size))
            print(self.net)

        if len(ts.data.shape) < 4:
            ts.data = ts.data[:, :, :, None]

        ### - - Training - -

        ## Per epoch:
        ##      - Training (with intermediate validation and affine correction
        ##        if active)
        ##      - Validation (full data)

        for epoch_idx in range(self.params.epochs):
            print('\nModel Optimization\n')

            ## (optional) Freeze MappingNet once we reach mapnet_freeze_epoch:
            # requires_grad=False stops its gradients (Adam then skips it), so
            # only the CNN (and the affine, if active) keep training.
            if (self.params.mapnet_freeze_epoch is not None and
                    not self._mapnet_frozen and
                    epoch_idx >= self.params.mapnet_freeze_epoch):
                for p in self.mapnet.parameters():
                    p.requires_grad_(False)
                self._mapnet_frozen = True
                print('\n- - - - - - - - - - - - - - - - - - - \n',
                      f'Epoch #{epoch_idx: 3d} - MappingNet weights frozen\n',
                      '- - - - - - - - - - - - - - - - - - - \n')

            ## (optional) Activate the per-frame affine correction once it is
            # due, adding its parameters as a new optimizer group (Adam state
            # for net/mapnet is preserved).
            if (self.params.affine_start_epoch is not None and
                    not self._affine_active and
                    epoch_idx >= self.params.affine_start_epoch):
                self.optimizer.add_param_group(
                    {'params': self.affine.parameters(),
                     'lr': self.params.affine_lr})
                self._affine_active = True
                n_train_batch = n_train
                print('\n- - - - - - - - - - - - - - - - - - - - - - - - - - ',
                      '- - - - - - - - - - - - \n',
                      f'Epoch #{self.step: 3d} - Affine correction enabled '
                      f'({self.params.frames} frames, ',
                      f'lr={self.params.affine_lr})\n',
                      '- - - - - - - - - - - - - - - - - - - - - - - - - - - ',
                      '- - - - - - - - - - - \n')

            ## Pair of frame and projection index: Keep track of the data already used
            if not self._affine_active:
                ## Phase 1 - no affine (fast)
                frame_idx_train = self.train_indices.repeat_interleave(self.params.proj_size).to(device=self.params.dev)
                depth_idx_train = torch.tile(torch.arange(self.params.proj_size), (n_train,)).to(device=self.params.dev)
                pair_array_train = torch.stack([frame_idx_train, depth_idx_train], dim=1).to(device=self.params.dev)  # shape: (n_train * self.params.proj_size, 2)
                perm_train = torch.randperm(n_train * self.params.proj_size)
                shuffled_pairs_train = pair_array_train[perm_train]
            else:
                ## Phase 2 - with affine (slow)
                frame_idx_train = self.train_indices.clone().to(device=self.params.dev)
                perm_train = torch.randperm(n_train)
                shuffled_frame_train = frame_idx_train[perm_train]

            for batch_idx in range(n_train_batch):
                self.optimizer.zero_grad()

                if not self._affine_active:
                    ## Phase 1 - no affine (fast)
                    i_batched_frames = shuffled_pairs_train[batch_idx*self.params.batch_size:(batch_idx+1)*self.params.batch_size, 0]
                    i_batched_depths = shuffled_pairs_train[batch_idx*self.params.batch_size:(batch_idx+1)*self.params.batch_size, 1]

                    data_1d_ref = torch.zeros([self.params.proj_size, self.params.batch_size, 1, self.params.output_nch]).to(self.params.dev)
                    for i in range(self.params.batch_size):
                        data_1d_ref[:, i, :, :] = ts.data[:, :, i_batched_frames[i], :]
                else:
                    ## Phase 2 - with affine (slow)
                    i_batched_frames = shuffled_frame_train[batch_idx]
                    i_batched_depths = self.params.proj_size

                    data_1d_ref = ts.data[:, :, i_batched_frames, :].clone()

                # Run forward
                out_set = self.reconstruct_slices(ts.angles[i_batched_frames].squeeze(), i_batched_depths, ts.times[i_batched_frames].squeeze())
                angles = ts.angles[i_batched_frames].squeeze()
                self.grad.X = out_set
                data_1d_rec = self.affine.warp(self.grad(angles), i_batched_frames)

                # Get TV
                tv_array = torch.permute(out_set, [3, 1, 2, 0])
                tv = L.total_variation(tv_array)

                # Backpropogate and update weights
                total_loss = (self.loss_fn(data_1d_rec, data_1d_ref) / self.params.batch_size +
                              (tv * self.params.noise_regularizer) +
                              (self._affine_active * self.params.affine_regularizer * self.affine.identity_penalty()))  # 0 if not active
                total_loss = total_loss.sum()
                total_loss.backward()

                if self.step % self.params.save_period == 0:
                    self.results._update_values('train', loss=[self.step, total_loss])
                    best_trainloss = self.save_model(os.path.join(save_path, self.params.wandb_name + '_train_loss.dip.pkl'), 'train_loss', 'min', return_score=True)

                ## Intermediate evaluation of the model
                intval_loss = None
                if self.params.intval_eval and self.step % self.params.intval_step == 0:
                    intval_loss = self._int_validation(ts, self.params.intval_duration)
                    self.results._update_values('intval', loss=[self.step, intval_loss], time=[self.step, time.time()-self.results.start])
                    
                    val_idx = torch.randint(0, len(self.val_indices), (1,), device=self.params.dev)[0]
                    metrics = self._evaluate(ts, eval_idx=val_idx.item(), title='intval')
                    print(f'Epoch #{epoch_idx: 3d} - Step #{self.step: 8d} - Loss = {intval_loss: 10.5g} - SSIM = {metrics["SSIM"]: 10.5g} - PSNR = {metrics["PSNR"]: 8.3g}')

                    best_intvalloss = self.save_model(os.path.join(save_path, self.params.wandb_name + '_intval_loss.dip.pkl'), 'intval_loss', 'min', return_score=True)
                    best_intvalssim = self.save_model(os.path.join(save_path, self.params.wandb_name + '_intval_SSIM.dip.pkl'), 'intval_SSIM', 'max', return_score=True)
                    best_intvalpsnr = self.save_model(os.path.join(save_path, self.params.wandb_name + '_intval_PSNR.dip.pkl'), 'intval_PSNR', 'max', return_score=True)

                if intval_loss is not None:
                    self.scheduler.step(intval_loss)

                self.optimizer.step()
                self.step += 1

            self._validation(ts, epoch_idx)

            best_fullvalloss = self.save_model(os.path.join(save_path, self.params.wandb_name + '_fullval_loss.dip.pkl'), 'fullval_loss', 'min', return_score=True)
            best_fullvalssim = self.save_model(os.path.join(save_path, self.params.wandb_name + '_fullval_SSIM.dip.pkl'), 'fullval_SSIM', 'max', return_score=True)
            best_fullvalpsnr = self.save_model(os.path.join(save_path, self.params.wandb_name + '_fullval_PSNR.dip.pkl'), 'fullval_PSNR', 'max', return_score=True)

            if self.params.hypertrain:
                pass
            else:
                self.results.publish()

        print(f'Best train loss: {best_trainloss[1]: 10.5g} at step #{int(best_trainloss[0]): 8d}')
        print(f'Best intermediate validation loss: {best_intvalloss[1]: 10.5g} at step #{int(best_intvalloss[0]): 8d}')
        print(f'Best intermediate validation SSIM: {best_intvalssim[1]: 10.5g} at step #{int(best_intvalssim[0]): 8d}')
        print(f'Best intermediate validation PSNR: {best_intvalpsnr[1]: 8.3g} at step #{int(best_intvalpsnr[0]): 8d}')
        print(f'Best full validation loss: {best_fullvalloss[1]: 10.5g} at epoch #{int(best_fullvalloss[0]): 4d}')
        print(f'Best full validation SSIM: {best_fullvalssim[1]: 10.5g} at epoch #{int(best_fullvalssim[0]): 4d}')
        print(f'Best full validation PSNR: {best_fullvalpsnr[1]: 8.3g} at epoch #{int(best_fullvalpsnr[0]): 4d}')


    @torch.no_grad()
    def _evaluate(self, ts, eval_idx=None, n_train=None, title=None):

        # Get reconstruction
        rec = np.zeros((self.params.proj_size, self.params.proj_size))

        if eval_idx is None:
            if n_train is None:
                idx = torch.randint(0, self.params.frames, (1,)).to(self.params.dev)
            else:
                rand_idx = torch.randint(0, n_train, (1,)).to(self.params.dev)
                idx = self.train_indices[rand_idx]
        else:
            idx = eval_idx

        if title is None:
            title = 'tiltseries'

        for i in range(self.params.proj_size):
            val = tomo.fp(self.reconstruct_slices(ts.angles[idx].squeeze(), i, ts.times[idx].squeeze()), ts.angles[idx].squeeze()).squeeze()
            rec[:, i] = util.torch_to_np(val)
        ref = ts.data[:, :, idx, :][:, :, 0, 0].squeeze()

        # Get metrics and log results on projections
        metrics = self.results.update(self.step, rec, ref, title)

        # Get metrics and log results on volumes
        if self.params.eval_vol:
            # reconstruct a slice of the volume at halfway through the centre and at time = 0
            rec = self.reconstruct_slices(ts.angles[0], self.params.proj_size//2, ts.times[0]).squeeze()
            rec = util.torch_to_np(rec).squeeze()

            # Get metrics on volume
            self.results.update(self.step, rec, self.ref_vol[:, self.params.proj_size//2, :].squeeze(), 'volume')

        return metrics

    @torch.no_grad()
    def _int_validation(self, ts, duration):

        n_val = len(self.val_indices)
        n_val_batch = (n_val * self.params.proj_size + self.params.batch_size - 1) // self.params.batch_size

        # f_idx_val = torch.repeat_interleave(self.val_indices, self.params.proj_size)
        f_idx_val = self.val_indices.repeat_interleave(self.params.proj_size).to(device=self.params.dev)
        d_idx_val = torch.tile(torch.arange(self.params.proj_size), (n_val,)).to(device=self.params.dev)
        pair_array_val = torch.stack([f_idx_val, d_idx_val], dim=1).to(device=self.params.dev)  # shape: (n_val * self.params.proj_size, 2)

        perm_val = torch.randperm(n_val * self.params.proj_size)
        shuffled_pairs_val = pair_array_val[perm_val]

        batch_val = 0
        intval_loss = []

        while batch_val < n_val_batch and batch_val < duration:
            i_batched_frames = shuffled_pairs_val[batch_val*self.params.batch_size:(batch_val+1)*self.params.batch_size, 0]
            i_batched_depths = shuffled_pairs_val[batch_val*self.params.batch_size:(batch_val+1)*self.params.batch_size, 1]

            out_set = self.reconstruct_slices(ts.angles[i_batched_frames].squeeze(), i_batched_depths, ts.times[i_batched_frames].squeeze())
            angles = ts.angles[i_batched_frames].squeeze()

            data_1d_ref = torch.zeros([self.params.proj_size, self.params.batch_size, 1, self.params.output_nch]).to(self.params.dev)
            for i in range(self.params.batch_size):
                data_1d_ref[:, i, :, :] = ts.data[:, :, i_batched_frames[i], :], i_batched_frames[i][:, i_batched_depths[i], 0, :][:, None, :]

            self.grad.X = out_set
            tv_array = torch.permute(out_set, [3, 1, 2, 0])
            tv = L.total_variation(tv_array)
            data_1d_rec = self.grad(angles)

            # Backpropogate and update weights
            temp_loss = self.loss_fn(data_1d_rec, data_1d_ref) + (tv*self.params.noise_regularizer)
            intval_loss += [(temp_loss / self.params.batch_size).sum().cpu(),]

            batch_val += 1

        return np.mean(np.array(intval_loss))

    @torch.no_grad()
    def _validation(self, ts, epoch_idx):

        fullval_loss = self._int_validation(ts, (len(self.val_indices) * self.params.proj_size + self.params.batch_size - 1) // self.params.batch_size)

        rec = np.zeros((len(self.val_indices), self.params.proj_size, self.params.proj_size))
        ref = ts.data[:, :, self.val_indices, :].squeeze()
        ref = ref.permute(2, 0, 1).cpu()

        val_ssim = []
        val_psnr = []
        for i, val_i in enumerate(self.val_indices):
            for proj_i in range(self.params.proj_size):
                val = tomo.fp(self.reconstruct_slices(ts.angles[val_i].squeeze(), proj_i, ts.times[val_i].squeeze()), ts.angles[val_i].squeeze()).squeeze()
                rec[i, :, proj_i] = util.torch_to_np(val)
            metrics = self.results.quantify(self.affine.warp(rec[i], i), ref[i])
            val_ssim += [metrics['SSIM'],]
            val_psnr += [metrics['PSNR'],]
        fullval_ssim = np.mean(np.array(val_ssim))
        fullval_psnr = np.mean(np.array(val_psnr))

        self.results._update_values('fullval', loss=[epoch_idx, fullval_loss], time=[epoch_idx, time.time()-self.results.start],
                                    SSIM=[epoch_idx, fullval_ssim], PSNR=[epoch_idx, fullval_psnr])

        rec[0] = (rec[0] - rec[0].min()) / (rec[0].max() - rec[0].min())
        ref[0] = (ref[0] - ref[0].min()) / (ref[0].max() - ref[0].min())

        self.results._update_images('fullval', [ref[0], rec[0]])

        print(f'Epoch #{epoch_idx: 3d} - Loss = {fullval_loss: 10.5g} - SSIM = {fullval_ssim: 10.5g} - PSNR = {fullval_psnr: 8.3g}')

    def reconstruct_slices(self, angles, depths, times):
        # Note all arrays must be same lenghth
        batch_val = 1
        if isinstance(depths, Iterable):
            if len(depths) > 1:
                batch_val = len(depths)
        manifold_output = self.manifold.get_value(angles, depths, times)
        mapnet_output = self.mapnet(manifold_output).reshape((batch_val, self.params.input_nch, self.params.style_size, self.params.style_size))
        out = self.net(mapnet_output)
        out = torch.permute(out, (2, 0, 3, 1))
        return out

    def reconstruct(self, times=None, angles=None, **kwargs):
        # Define kwargs ts is a tiltseries object
        # or you can just input the angles and times

        usesetangle = kwargs.get('usesetangle', False)
        ts = kwargs.get('ts', None)

        if times is None and angles is None:
            angles = kwargs.get('angles', ts.angles)
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
                rec[:, :, :] = util.torch_to_np(self.reconstruct_slices(tiled_angles, depths, tiled_times)).squeeze()

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
        except (AttributeError, RuntimeError) as e:
            logging.warning("state_dict() failed: %s", e)
            raise

    def save_model(self, path, title, mode, return_score=True):

        try:            # Safely convert any nested CUDA tensors to CPU/Python scalars before numpy conversion
            tables_data = self.results.tables[title]

            def _to_numpy_safe(obj):
                if isinstance(obj, torch.Tensor):
                    # Detach from graph and move to CPU. Scalars become Python floats, others stay as numpy arrays.
                    return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().numpy()
                if isinstance(obj, (list, tuple)):
                    return type(obj)(_to_numpy_safe(x) for x in obj)
                return obj

            tables_data = _to_numpy_safe(tables_data)
            values_arr = np.array(tables_data)
        except KeyError as err:
            raise KeyError('This key is not tracked during the training') from err

        if mode == 'min':
            best_score_idx = np.argmin(values_arr[:, 1], axis=0)
            best_score = values_arr[best_score_idx]
            if values_arr[-1, 1] == best_score[1]:
                torch.save(self.state_dict(), path)
        elif mode == 'max':
            best_score_idx = np.argmax(values_arr[:, 1], axis=0)
            best_score = values_arr[best_score_idx]
            if values_arr[-1, 1] == best_score[1]:
                torch.save(self.state_dict(), path)
        else:
            raise ValueError("Save_model mode is not 'min' or 'max'")

        if return_score:
            return best_score

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
        solver.params.eval_vol = False
        solver.params.hypertrain = False

        return solver
