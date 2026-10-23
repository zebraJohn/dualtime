import os
import torch
import numpy as np
from models import Times2D,TimesNet, PatchTST, Autoformer, Transformer, DLinear
from models import PhaseTimes2D, DualTimes2D, PhaseFormer, DualPhaseFormer
from models import FreDN, DualFreDN, CrossChannelFreq
from models import FreEformer, DualFreEformer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'TimesNet' : TimesNet,
            'Times2D' :Times2D,
            'PhaseTimes2D': PhaseTimes2D,
            'DualTimes2D': DualTimes2D,
            'PhaseFormer': PhaseFormer,
            'DualPhaseFormer': DualPhaseFormer,
            'FreDN': FreDN,
            'DualFreDN': DualFreDN,
            'CrossChannelFreq': CrossChannelFreq,
            'FreEformer': FreEformer,
            'DualFreEformer': DualFreEformer}

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
            if self.args.use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print(f'device = {device}')
                print('Use GPU: cuda:{}'.format(self.args.gpu))
            else:
                device = torch.device('cpu')
                print('Use CPU')
                
            return device

    
    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
