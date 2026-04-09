# Dataset Creation Scripts

These scripts create the BOLD5000-QA dataset from raw BOLD5000 fMRI data. This is a two-step pipeline:

## Step 1: Generate Scene Graphs

Extract scene graphs from BOLD5000 images using an LLM (requires OpenAI API key):

```bash
cd scripts/dataset
source create_bold5000_scene_graph.sh
```

This calls `src/create_dataset/create_bold5000_scene_graph.py`, which:
- Loads BOLD5000 images from WebDataset tar files
- Sends images to an LLM to generate structured scene graphs (objects, attributes, relations)
- Saves scene graphs to `data/bold5000_scene_graph.csv`

## Step 2: Generate FQA Dataset

Convert scene graphs into fMRI question-answer pairs:

```bash
cd scripts/dataset
source create_bold5000_fqa.sh <subject>
```

For example:
```bash
source create_bold5000_fqa.sh CSI1
```

This calls `src/create_dataset/create_bold5000_fqa.py`, which:
- Loads the scene graph CSV and BOLD5000 fMRI data
- Generates compositional queries (exists, descriptor, relation) from scene graphs
- Pairs queries with fMRI brain activity and ground-truth answers
- Saves per-image `.npy` files to `data/BOLD5000-QA/<subject>/{train,test}/`

## Prerequisites

- Raw BOLD5000 data in WebDataset format (see `scripts/download_data.sh`)
- OpenAI API key (for scene graph generation only)
- Activated `neurona` conda environment
