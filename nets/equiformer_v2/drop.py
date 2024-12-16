'''
    Add `extra_repr` into DropPath implemented by timm 
    for displaying more info.
'''


import torch
import torch.nn as nn
from e3nn import o3
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'drop_prob={}'.format(self.drop_prob)
    
    
class GraphDropPath(nn.Module):
    '''
        Consider batch for graph data when dropping paths.
    '''
    def __init__(self, drop_prob=None):
        super(GraphDropPath, self).__init__()
        self.drop_prob = drop_prob
        

    def forward(self, x, batch):
        batch_size = batch.max() + 1
        shape = (batch_size,) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)
        drop = drop_path(ones, self.drop_prob, self.training)
        out = x * drop[batch]
        return out
    
    
    def extra_repr(self):
        return 'drop_prob={}'.format(self.drop_prob)
    
    

class EquivariantDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        super(EquivariantDropout, self).__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(irreps, 
            o3.Irreps('{}x0e'.format(self.num_irreps)))
        
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0], self.num_irreps)
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        out = self.mul(x, mask)
        return out
    

class EquivariantScalarsDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        super(EquivariantScalarsDropout, self).__init__()
        self.irreps = irreps
        self.drop_prob = drop_prob
        
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        out = []
        start_idx = 0
        for mul, ir in self.irreps:
            temp = x.narrow(-1, start_idx, mul * ir.dim)
            start_idx += mul * ir.dim
            if ir.is_scalar():
                temp = F.dropout(temp, p=self.drop_prob, training=self.training)
            out.append(temp)
        out = torch.cat(out, dim=-1)
        return out
    
    
    def extra_repr(self):
        return 'irreps={}, drop_prob={}'.format(self.irreps, self.drop_prob)
    

class EquivariantDropoutArraySphericalHarmonics(nn.Module):
    def __init__(self, drop_prob, drop_graph=False):
        super(EquivariantDropoutArraySphericalHarmonics, self).__init__()
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.drop_graph = drop_graph
        
        
    def forward(self, x, batch=None):
        if not self.training or self.drop_prob == 0.0:
            return x
        assert len(x.shape) == 3

        if self.drop_graph:
            assert batch is not None
            batch_size = batch.max() + 1
            shape = (batch_size, 1, x.shape[2])
            mask = torch.ones(shape, dtype=x.dtype, device=x.device)
            mask = self.drop(mask)
            out = x * mask[batch]
        else:
            shape = (x.shape[0], 1, x.shape[2])
            mask = torch.ones(shape, dtype=x.dtype, device=x.device)
            mask = self.drop(mask)
            out = x * mask

        return out
    
    
    def extra_repr(self):
        return 'drop_prob={}, drop_graph={}'.format(self.drop_prob, self.drop_graph)
    
    
##########################################################################################
# Change@DEQ: Recurrent / Variational Dropout
# applies the same mask at every solver step 
# inspired by:
# Deep Equilibrium Models https://arxiv.org/abs/1909.01377
# Code: https://github.com/locuslab/deq/blob/1fb7059d6d89bb26d16da80ab9489dcc73fc5472/lib/optimizations.py
##########################################################################################


class RecurrentGraphPathDrop(nn.Module):
    """
    Sets some batches (all atoms in a molecule) to zero and rescales the others to 1/1-p.
    If used with residual connections, only the residual (input) will be kept.

    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
        self.mask = None

        # self.update_mask()
        # self.register_buffer("mask", self.mask)

    def forward(self, x, batch):
        """Sets some batches (all atoms in a molecule) to zero and rescales the others to 1/1-p."""
        # for batch_size=4, 21 atoms per molecule
        # batch: tensor([
        # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
        # 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        # 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        batch_size = batch.max() + 1
        shape = (batch_size,) + (1,) * (x.ndim - 1)
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)
        # drop = self.path_drop(ones, self.drop_prob, self.training)
        if self.drop_prob == 0.0 or not self.training:
            drop = ones
        else:
            keep_prob = 1 - self.drop_prob
            # div: rescale to 1/ 1-probability
            drop = ones.div(keep_prob) * self.mask
        # drop: [batch_size, 1, 1]. some batches are 0, other are 1/1-p
        # drop[batch]: [batch_size*num_atoms, 1, 1]. some batches are 0, other are 1/1-p
        out = x * drop[batch]
        return out

    # def forward(self, x, batch):
    #     """Sets some batches (all atoms in a molecule) to zero and rescales the others to 1/1-p."""
    #     # for batch_size=4, 21 atoms per molecule
    #     # batch: tensor([
    #     # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    #     # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
    #     # 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    #     # 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    #     batch_size = batch.max() + 1
    #     shape = (batch_size,) + (1,) * (x.ndim - 1)
    #     ones = torch.ones(shape, dtype=x.dtype, device=x.device)
    #     drop = self.path_drop(ones, self.drop_prob, self.training)
    #     # drop: [batch_size, 1, 1]. some batches are 0, other are 1/1-p
    #     # drop[batch]: [batch_size*num_atoms, 1, 1]. some batches are 0, other are 1/1-p
    #     out = x * drop[batch]
    #     return out

    def path_drop(self, x, drop_prob: float = 0.0, training: bool = False):
        """Same as the path_drop function above but with persistent mask."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # div: rescale to 1/ 1-probability
        output = x.div(keep_prob) * self.mask
        return output

    def update_mask(self, x, batch):
        """Sets random tensor for which batches to drop.
        Call in the beginning of each forward pass of the parent model.
        """
        # work with diff dim tensors, not just 2D ConvNets
        batch_size = batch.max() + 1
        shape = (batch_size,) + (1,) * (x.ndim - 1)
        keep_prob = 1 - self.drop_prob
        mask = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device, requires_grad=False
        )
        mask.floor_()  # binarize
        self.mask = mask
        return mask

    def extra_repr(self):
        return "drop_prob={}".format(self.drop_prob)


class RecurrentDropout(nn.Module):
    """Applies the same droput mask at every solver step.

    The name comes from the inspiration through Baysian / variational inference.
    A Theoretically Grounded Application of Dropout in Recurrent Neural Networks

    Code is based on:
    https://github.com/locuslab/deq/blob/1fb7059d6d89bb26d16da80ab9489dcc73fc5472/lib/optimizations.py#L145

    Something else: Variational dropout (VD) is a generalization of Gaussian dropout, which aims at inferring the posterior of network weights based on a log-uniform prior on them to learn
    these weights as well as dropout rate simultaneously.
    """

    # See also other ways to implement:
    # https://github.com/mourga/variational-lstm
    # https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py
    # Applies the same dropout mask across the temporal dimension,
    # but not applied to the recurrent activations in the LSTM like https://arxiv.org/abs/1512.05287

    def __init__(self, drop_prob=0.0, length_first=False):
        """
        Hidden-to-hidden (VD-based) dropout that applies the same mask at every time step and every layer
        :param dropout: The dropout rate (0 means no dropout is applied)
        :param temporal: Whether the dropout mask is the same across the temporal dimension (or only the depth dimension)
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.mask = None
        self.length_first = length_first

    def update_mask(self, shape, dtype, device, set_to_none=False):
        if set_to_none:
            # if we don't yet know the shape, we can set the mask to None
            self.mask = None
            return
        else:
            # Dimension (N, C, L)
            # m = torch.zeros(bsz, d, 1).bernoulli_(1 - self.drop_prob)
            m = torch.zeros(
                shape, dtype=dtype, device=device, requires_grad=False
            ).bernoulli_(1 - self.drop_prob)
            # rescale to 1/ 1-probability
            mask = m / (1 - self.drop_prob)
            mask.requires_grad = False
            self.mask = mask
            return mask

    def forward(self, x):
        if (not self.training) or (self.drop_prob == 0):
            return x
        # generate mask on the fly and save it
        # if self.mask is None:
        #     self.update_mask(x.shape, x.device, set_to_none=False)
        assert self.mask is not None, "Mask is not initialized"
        mask = self.mask.expand_as(x)  # Make sure the dimension matches
        return mask * x
