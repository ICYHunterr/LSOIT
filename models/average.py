import argparse
import collections
import torch
import os
import re

def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.
    Args:
      inputs: An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    #first = True
    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state

        model_params_keys = list(model_params.keys())
        if params_keys is None: #and first:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                #if not first: continue
                params_dict[k] = p.clone()
                #params_dict[k] = p.clone().zero_()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
                #params_dict[k] = torch.max(torch.cat([p.unsqueeze(dim=0), params_dict[k].unsqueeze(dim=0)], dim=0), dim=0)[0]
        #first = False

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        averaged_params[k].div_(num_models)
    new_state = averaged_params
    return new_state
