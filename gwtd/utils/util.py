import sys
import numpy
import torch
import random


def setup_seed(seed: int = 0):
    """
    Setup random seed and deterministic behavior.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def export_onnx(args):
    import onnx  # noqa

    inputs = ['images']
    outputs = ['outputs']
    dynamic = {'outputs': {0: 'batch', 1: 'anchors'}}

    m = torch.load('./weights/best.pt')['model'].float()
    x = torch.zeros((1, 3, args.input_size, args.input_size))

    torch.onnx.export(m.cpu(), x.cpu(),
                      f='./weights/best.onnx',
                      verbose=False,
                      opset_version=12,
                      # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
                      do_constant_folding=True,
                      input_names=inputs,
                      output_names=outputs,
                      dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load('./weights/best.onnx')  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    onnx.save(model_onnx, './weights/best.onnx')
    # Inference example
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py


def load_weight(model, ckpt_path):
    """
    Load weights into model from a checkpoint file that may contain either:
    - a full model object under key 'model' (Ultralytics style), or
    - a state_dict under key 'model' (this project's best.pt), or
    - a raw state_dict at the top level.
    Mismatched shapes/keys are skipped.
    """
    dst_state = model.state_dict()

    # Register old bare module names as aliases so torch.load can unpickle
    # weight files that were saved under the original module paths.
    import gwtd.nets as _nets_pkg
    import gwtd.nets.nn as _nn_mod
    sys.modules.setdefault('nets', _nets_pkg)
    sys.modules.setdefault('nets.nn', _nn_mod)

    obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Extract source state_dict from various formats
    if isinstance(obj, dict) and 'model' in obj:
        src_obj = obj['model']
    else:
        src_obj = obj

    if isinstance(src_obj, torch.nn.Module):
        src_state = src_obj.float().cpu().state_dict()
    elif isinstance(src_obj, (dict,)):
        src_state = src_obj
    else:
        raise TypeError(f"Unsupported checkpoint format in {ckpt_path}: {type(src_obj)}")

    # Filter compatible keys
    filtered = {k: v for k, v in src_state.items() if k in dst_state and v.shape == dst_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    print(f'Loaded weights from {ckpt_path}')
    return

def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)