import torch
import numpy as np

class AbstractTensorizer():
    def __init__(self, device='cpu'):
        self.device = device

    def tensorize(self, observation):
        pass