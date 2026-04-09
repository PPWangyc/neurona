import argparse
import json
import os
import random

import numpy as np
import torch
import yaml

from utils.log_utils import logger


def get_args():
    parser = argparse.ArgumentParser(description='NEURONA: Neuro-Symbolic Decoding of Neural Activity')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--subject', type=str, default=None, help='Subject (overrides config)')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--train_hop', type=str, default=None, help='Training hop: all, zs, 1, 2, or 3')
    parser.add_argument('--atlas', type=str, default=None, help='Brain atlas: yeo7, yeo17, difumo1024, whole')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--model', type=str, default=None, help='Model name')
    parser.add_argument('--ground_attr_region', action='store_true', default=None, help='Ground attribute concepts to brain regions')
    parser.add_argument('--ground_attr_relation', action='store_true', default=None, help='Ground attribute concepts using region relations')
    parser.add_argument('--ground_rel_region', action='store_true', default=None, help='Ground relational concepts to brain regions')
    parser.add_argument('--ground_rel_relation', action='store_true', default=None, help='Ground relational concepts using region relations')
    parser.add_argument('--ground_rel_guided_relation', type=str, default=None, help='Guidance mode for relation grounding')
    parser.add_argument('--plot_example_images', action='store_true', default=None, help='Plot example images')
    parser.add_argument('--removed_regions', type=str, nargs='+', default=None, help='Atlas region names to remove for ablation')
    parser.add_argument('--selected_regions', type=str, nargs='+', default=None, help='Atlas region names to select for encoding')
    cli_args = parser.parse_args()

    # Defaults
    defaults = {
        'data_dir': 'data',
        'dataset': 'BOLD5000',
        'subject': 'CSI1',
        'epochs': 200,
        'batch_size': 32,
        'seed': 42,
        'train_hop': 'all',
        'atlas': 'yeo17',
        'output_dir': 'output',
        'model': 'ns',
        'ground_attr_region': False,
        'ground_attr_relation': False,
        'ground_rel_region': False,
        'ground_rel_relation': False,
        'ground_rel_guided_relation': 'off',
        'plot_example_images': False,
        'removed_regions': None,
        'selected_regions': None,
    }

    # Load YAML config if provided
    if cli_args.config is not None:
        with open(cli_args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        defaults.update(yaml_config)

    # CLI args override config (only non-None values)
    for key, value in vars(cli_args).items():
        if key == 'config':
            continue
        if value is not None:
            defaults[key] = value

    args = argparse.Namespace(**defaults)
    return args

def set_seed(seed):
    # set seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed set to {}'.format(seed))

def move_batch_to_device(batch, device):
    # if batch values are tensors, move them to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch
def process_args(args, make_dir=True) -> str:
    """Create and return the output directory path based on the args.
    
    Args:
        args: Argument object containing all necessary parameters
        make_dir: Whether to create the output directory (default: True)
        
    Returns:
        str: Path to the created output directory
    """
    # Build attr_ground and rel_ground strings
    attr_ground = ""
    rel_ground = ""
    dataset = args.dataset
    if args.ground_attr_region:
        attr_ground += "-region"
    if args.ground_attr_relation:
        attr_ground += "-relation"
    if args.ground_rel_region:
        rel_ground += "-region"
    rel_ground += f"-{args.ground_rel_guided_relation}"
    if args.ground_rel_relation:
        rel_ground += "-relation"
    
    assert attr_ground != "" or rel_ground != "", "No grounding is selected"
    
    # Add removed regions or selected regions suffix if provided
    regions_suffix = ""
    if hasattr(args, 'selected_regions') and args.selected_regions is not None and len(args.selected_regions) > 0:
        # Create a sorted, joined string of selected regions for directory naming
        selected_regions_str = "_".join(sorted(args.selected_regions))
        regions_suffix = f"_selected-{selected_regions_str}"
    elif hasattr(args, 'removed_regions') and args.removed_regions is not None and len(args.removed_regions) > 0:
        # Create a sorted, joined string of removed regions for directory naming
        removed_regions_str = "_".join(sorted(args.removed_regions))
        regions_suffix = f"_removed-{removed_regions_str}"
    
    # Construct the output directory path
    output_dir = os.path.join(
        args.output_dir,
        f"ds-{dataset}",
        f"sub-{args.subject}",
        f"model-ns",
        f"atlas-{args.atlas}{regions_suffix}",
        f"attr{attr_ground}",
        f"rel{rel_ground}",
        f"hop-{args.train_hop}",
    )

    # set a dict to store the args
    args_dict = {
        'attr_ground': attr_ground,
        'rel_ground': rel_ground,
        'train_hop': args.train_hop,
        'atlas': args.atlas,
        'ground_attr_region': args.ground_attr_region,
        'ground_attr_relation': args.ground_attr_relation,
        'ground_rel_region': args.ground_rel_region,
        'ground_rel_relation': args.ground_rel_relation,
        'ground_rel_multi_relation': False,
        'ground_rel_guided_relation': args.ground_rel_guided_relation,
        'output_dir': output_dir,
        'dataset': dataset,
        'subject': args.subject,
    }
    # Add selected_regions or removed_regions if provided
    if hasattr(args, 'selected_regions') and args.selected_regions is not None:
        args_dict['selected_regions'] = args.selected_regions
    if hasattr(args, 'removed_regions') and args.removed_regions is not None:
        args_dict['removed_regions'] = args.removed_regions
    # Create the directory if it doesn't exist
    if make_dir:
        os.makedirs(output_dir, exist_ok=True)
        # save args_dict to output_dir/args.json
        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            json.dump(args_dict, f)
    
    return args_dict

def load_best_model(model: torch.nn.Module, output_dir: str, train_hop: str, atlas: str) -> torch.nn.Module:
    """Load the best model from the output directory.
    
    Args:
        model (torch.nn.Module): The model to load the weights into
        output_dir (str): Path to the output directory
        train_hop (str): Training hop value
        atlas (str): Atlas name
        
    Returns:
        torch.nn.Module: The model with loaded weights
    """
    model_path = os.path.join(output_dir, f'best_model_{train_hop}_{atlas}.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}")
    
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

