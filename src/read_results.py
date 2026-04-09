import glob
import os
import re

from utils.plot_utils import log_evaluation_results
from utils.utils import get_args, process_args, set_seed


def main(args):
    dataset_name = args.dataset
    model_name = args.model
    output_dir = args.output_dir
    train_hop = args.train_hop
    removed_regions = getattr(args, 'removed_regions', None)
    selected_regions = getattr(args, 'selected_regions', None)
    
    files = glob.glob(os.path.join(output_dir, '**', '*.pth'), recursive=True)
    files = [os.path.abspath(file) for file in files]
    res_dirs = [os.path.dirname(file) for file in files]
    
    # filter out the files that do not contain the dataset name
    res_dirs = [res_dir for res_dir in res_dirs if f'ds-{dataset_name}' in res_dir]
    # filter out the files that do not contain the train_hop
    res_dirs = [res_dir for res_dir in res_dirs if f'hop-{train_hop}' in res_dir]
    # filter out the files that do not contain the model name
    res_dirs = [res_dir for res_dir in res_dirs if f'model-{model_name}' in res_dir]
    
    # Filter by selected_regions or removed_regions if provided
    if selected_regions is not None and len(selected_regions) > 0:
        # Create the selected regions pattern (sorted and joined with underscores, matching process_args)
        selected_regions_str = "_".join(sorted(selected_regions))
        selected_regions_pattern = f"_selected-{selected_regions_str}"
        print(f"Filtering for selected regions: {selected_regions}")
        print(f"Looking for pattern: {selected_regions_pattern}")
        # Filter directories that contain the selected regions pattern
        res_dirs = [res_dir for res_dir in res_dirs if selected_regions_pattern in res_dir]
    elif removed_regions is not None and len(removed_regions) > 0:
        # Create the removed regions pattern (sorted and joined with underscores, matching process_args)
        removed_regions_str = "_".join(sorted(removed_regions))
        removed_regions_pattern = f"_removed-{removed_regions_str}"
        print(f"Filtering for removed regions: {removed_regions}")
        print(f"Looking for pattern: {removed_regions_pattern}")
        # Filter directories that contain the removed regions pattern
        res_dirs = [res_dir for res_dir in res_dirs if removed_regions_pattern in res_dir]
    else:
        # If no regions specified, filter out any directories with removed or selected regions
        # (to only get baseline results without ablation)
        res_dirs = [res_dir for res_dir in res_dirs if '_removed-' not in res_dir and '_selected-' not in res_dir]
    
    model_args = process_args(args, make_dir=False)
    
    # For selected_regions or removed_regions queries, we want to search across all subjects
    # So we don't filter by subject if regions are specified
    if (selected_regions is not None and len(selected_regions) > 0) or (removed_regions is not None and len(removed_regions) > 0):
        # Only filter by atlas and other configs, but not by subject
        # Build the expected path pattern without subject
        # Pattern should be: atlas-{atlas}_selected-{regions} or atlas-{atlas}_removed-{regions}/attr{...}/rel{...}/hop-{...}
        atlas_name = getattr(args, 'atlas', 'yeo17')
        # The regions pattern is already filtered above
        # Now filter by other configs (attr, rel, hop) but allow any subject
        # Extract the config part from model_args (everything after subject)
        model_dir_full = model_args['output_dir']
        # Remove dataset, model, and output_dir base
        model_dir = model_dir_full.replace(f'ds-{dataset_name}', '').replace(f'model-{model_name}', '').replace(output_dir, '')
        # Remove any subject pattern (sub-*) to allow matching across subjects
        model_dir = re.sub(r'sub-[^/]+/?', '', model_dir)
        model_dir = model_dir.strip('/')
        # Filter by the remaining model_dir pattern (atlas with regions, attr, rel, hop)
        if model_dir:
            res_dirs = [res_dir for res_dir in res_dirs if model_dir in res_dir]
    else:
        # Normal filtering with subject when no regions specified
        model_dir = model_args['output_dir']
        model_dir = model_dir.replace(f'ds-{dataset_name}', '').replace(f'model-{model_name}', '').replace(output_dir, '').replace(f'sub-{args.subject}', '')
        model_dir = model_dir.strip('/')
        if model_dir:
            res_dirs = [res_dir for res_dir in res_dirs if model_dir in res_dir]
    
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    if selected_regions:
        print(f"Selected regions: {selected_regions}")
        print(f"Searching across all subjects")
    elif removed_regions:
        print(f"Removed regions: {removed_regions}")
        print(f"Searching across all subjects")
    else:
        print(f"Subject: {args.subject}")
    print(f"Found {len(res_dirs)} result directories")
    
    if len(res_dirs) == 0:
        print("Warning: No matching result directories found!")
        return
    
    log_evaluation_results(res_dirs)
    
if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    main(args)