import os
import sys
import tqdm
import yaml
import torch
from argparse import ArgumentParser


# Add project root to Python path for model loading
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gwtd.nets import nn
from gwtd.utils import training_utils


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # load config
    config_path = os.path.join(project_root, 'config', 'default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    weight_decay = config['weight_decay']
    min_lr = config['min_lr']
    momentum = config['momentum']

    # create model without head
    name_backbone = config['backbone']
    pretrained_weights_path = os.path.join(project_root, 'weights', f'{name_backbone}.pt')
    model_with_custom_head = nn.YOLOwithCustomHead(name_backbone,
                                                   pretrained_weights_path,
                                                   config['network']['input_image_shape'],
                                                   from_logits=config['from_logits'])
    model_with_custom_head.cuda()

    # forward a dummy tensor
    x = torch.randn(1, 3, config['network']['input_image_shape'][0],
                    config['network']['input_image_shape'][1]).cuda()
    out = model_with_custom_head(x)
    # print the type of out
    print(type(out))
    for i in out:
        print(i.shape)

    # Optimizer
    optimizer = torch.optim.SGD(training_utils.set_params(model_with_custom_head, weight_decay),
                                min_lr, momentum, nesterov=True)


if __name__ == "__main__":
    main()
