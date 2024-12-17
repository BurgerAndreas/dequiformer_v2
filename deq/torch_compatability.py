import torch 

# for compatability: older torch versions do not support @torch.compiler.disable
# AttributeError: module 'torch' has no attribute 'compile'
# so define a dummy decorator if torch does not have the attribute
if not hasattr(torch, "compiler"):
    class DummyCompiler:
        @staticmethod
        def disable(recursive=True):
            return lambda f: f
    torch.compiler = DummyCompiler()


import inspect
def get_collate(dataset, follow_batch=None, exclude_keys=None):
    # torch geometric 2.2 and 2.4 have different call signatures
    if len(inspect.signature(Collater).parameters) == 3:
        # 2.4 https://github.com/pyg-team/pytorch_geometric/blob/b823c7e80d9396ee5095c2ef8cea0475d8ce7945/torch_geometric/loader/dataloader.py#L13
        return Collater(dataset, follow_batch=follow_batch, exclude_keys=exclude_keys)
    elif len(inspect.signature(Collater).parameters) == 2:
        # 2.2 https://github.com/pyg-team/pytorch_geometric/blob/ca4e5f8e308cf4ee7a221cb0979bb12d9e37a318/torch_geometric/loader/dataloader.py#L11
        return Collater(follow_batch=follow_batch, exclude_keys=exclude_keys)
    else:
        raise ValueError("Unknown Collater signature")


# torch_geometric/data/collate.py:145: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
# storage = elem.storage()._new_shared(numel)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric.data.collate')
