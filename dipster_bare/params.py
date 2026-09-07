import os
import sys
import numpy as np
import torch
import logging
class Params():
    def __init__(self, ts=None):
        # Data Parameters
        if ts is not None:
            self.proj_size: int = ts.data.shape[0]
            self.frames: int = ts.data.shape[2]
            self.noise_regularizer = 0.0001
            self.dev = 0
            self.seed: int = 0

            # Model Parameters
            self.opt_over: str = "net"
            self.latent_dim: int = 4
            self.hidden_dim: int = 512
            self.style_size: int = 8
            self.depth: int = 1
            self.Nr: int = 1
            self.input_nch: int = 1
            self.output_nch: int = 1
            self.need_bias: bool = False
            self.up_factor: int = 16
            self.upsample_mode: str = "nearest"
            self.output_activation: str = 'none' # 'none' (unbounded, default), 'sigmoid' -> (0,1), 'hardtanh' -> [0,1].

            # Training Parameters
            self.lr: float = 1e-3
            self.step_size: int = 2000
            self.gamma: float = 0.5
            self.batch_size: int = 1
            self.max_steps: int = 2000
            self.mapnet_freeze_step = None   # step to freeze MappingNet weights; None => never

            # Affine forward-model correction (per-frame 2x3 affine; off by default)
            self.affine_start_step = None       # step to switch the affine on; None => never
            self.affine_lr: float = 1e-3        # lr for the affine param group
            self.affine_regularizer: float = 1.0  # weight on identity_penalty()
            self.affine_frames_per_step: int = 1  # #full-frame projections per phase-2 step

            # torch.compile the CNN/MLP submodules
            self.compile_net: bool = False
            self.compile_mode: str = 'default'

            # Testing Parameters
            self.wandb_local_dir = r"..\..\wandb"
            self.wandb_project = 'Default'
            self.wandb_name = 'Default'
            self.evaluate: bool = True
            self.eval_vol: bool = False
            self.save_period: int = 100
            self.hypertrain: bool = False

    def set_env(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.dev)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def to_dict(self):
        _dict = {}
        _dict['proj_size'] = self.proj_size
        _dict['frames'] = self.frames
        _dict['dev'] = self.dev
        _dict['seed'] = self.seed
        _dict['opt_over'] = self.opt_over
        _dict['latent_dim'] = self.latent_dim
        _dict['hidden_dim'] = self.hidden_dim
        _dict['style_size'] = self.style_size
        _dict['depth'] = self.depth
        _dict['Nr'] = self.Nr
        _dict['input_nch'] = self.input_nch
        _dict['output_nch'] = self.output_nch
        _dict['need_bias'] = self.need_bias
        _dict['up_factor'] = self.up_factor
        _dict['upsample_mode'] = self.upsample_mode
        _dict['output_activation'] = self.output_activation
        _dict['lr'] = self.lr
        _dict['step_size'] = self.step_size
        _dict['gamma'] = self.gamma
        _dict['batch_size'] = self.batch_size
        _dict['max_steps'] = self.max_steps
        _dict['affine_start_step'] = self.affine_start_step
        _dict['affine_lr'] = self.affine_lr
        _dict['affine_regularizer'] = self.affine_regularizer
        _dict['affine_frames_per_step'] = self.affine_frames_per_step
        _dict['compile_net'] = self.compile_net
        _dict['compile_mode'] = self.compile_mode
        _dict['evaluate'] = self.evaluate
        _dict['eval_vol'] = self.eval_vol
        _dict['save_period'] = self.save_period
        _dict['wandb_local_dir'] = self.wandb_local_dir
        _dict['wandb_project'] = self.wandb_project
        _dict['wandb_name'] = self.wandb_name
        _dict['hypertrain'] = self.hypertrain
        _dict['mapnet_freeze_step'] = self.mapnet_freeze_step
        return _dict

    def log(self):
        #set up logging in juypter notebook
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', stream=sys.stdout)
        # logging.basicConfig(level=logging.NOTSET, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', stream=sys.stdout)

        dict = self.to_dict()
        for dict_key in dict:
            logging.info(f"{dict_key}: {dict[dict_key]}")


    @classmethod
    def from_dict(cls, state_dict):
        params = cls()
        params.proj_size = state_dict['proj_size']
        params.frames = state_dict['frames']
        params.dev = state_dict['dev']
        params.seed = state_dict['seed']
        params.opt_over = state_dict['opt_over']
        params.latent_dim = state_dict['latent_dim']
        params.hidden_dim = state_dict['hidden_dim']
        params.style_size = state_dict['style_size']
        params.depth = state_dict['depth']
        params.Nr = state_dict['Nr']
        params.input_nch = state_dict['input_nch']
        params.output_nch = state_dict['output_nch']
        params.need_bias = state_dict['need_bias']
        params.up_factor = state_dict['up_factor']
        params.upsample_mode = state_dict['upsample_mode']
        # .get() so checkpoints saved before this feature still load.
        params.output_activation = state_dict.get('output_activation', 'none')
        params.lr = state_dict['lr']
        params.step_size = state_dict['step_size']
        params.gamma = state_dict['gamma']
        params.batch_size = state_dict['batch_size']
        params.max_steps = state_dict['max_steps']
        # Affine fields: .get() so checkpoints saved before this feature still load.
        params.affine_start_step = state_dict.get('affine_start_step', None)
        params.affine_lr = state_dict.get('affine_lr', 1e-3)
        params.affine_regularizer = state_dict.get('affine_regularizer', 1.0)
        params.affine_frames_per_step = state_dict.get('affine_frames_per_step', 1)
        params.compile_net = state_dict.get('compile_net', False)
        params.compile_mode = state_dict.get('compile_mode', 'default')
        params.evaluate = state_dict['evaluate']
        params.eval_vol = state_dict['eval_vol']
        params.save_period = state_dict['save_period']
        params.wandb_local_dir = state_dict['wandb_local_dir']
        params.wandb_project = state_dict['wandb_project']
        params.wandb_name = state_dict['wandb_name']
        params.hypertrain = state_dict['hypertrain']
        params.mapnet_freeze_step = state_dict.get('mapnet_freeze_step', None)
        return params
