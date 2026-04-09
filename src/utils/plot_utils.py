import json
import logging
import os
from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from models.left.domain import create_domain_from_query, read_concepts_v2
from utils.dataset_utils import (compute_systematicity_accuracy,
                                 report_systematicity_accuracy,
                                 symbolic_to_natural_language)
from utils.log_utils import logger


def plot_bar_chart(
        data_dict, 
        keys, 
        names, 
        xlabel='',
        ylabel='',
        title='',
        colors=None,
        fontsize=12,
        linewidth=1.5,
        bar_width=1
    ):
    if colors is None:
        colors = sns.color_palette("flare", len(keys))
    # Filter dictionary and get values in the same order as keys
    values = [data_dict[k] for k in keys]
    
    # Convert values to K format if they're large
    max_value = max(values)
    if max_value >= 1000:
        values = [v/1000 for v in values]
        if 'Count' in ylabel or 'Number' in ylabel:
            ylabel = ylabel.replace('Count', 'Count (K)').replace('Number', 'Number (K)')
        else:
            ylabel = f"{ylabel} (K)"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # set labels as names
    bars = ax.bar(
        names, 
        values, 
        color=colors, 
        # width=bar_width
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}K' if max_value >= 1000 else f'{height:.0f}',
                ha='center', va='bottom', fontsize=fontsize-2)
    
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # set linewidth
    for spine in ax.spines:
        ax.spines[spine].set_linewidth(linewidth)
    return fig, ax

def plot_pie_chart(
        data_dict, 
        keys, 
        names, 
        title='',
        colors=None,
        fontsize=12
    ):
    if colors is None:
        colors = sns.color_palette("Set2", len(keys))
    # Filter dictionary and get values in the same order as keys
    values = [data_dict[k] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 6))
    # set labels as names
    ax.pie(
        values, 
        labels=names, 
        autopct='%1.1f%%', 
        colors=colors,
        textprops={'fontsize': fontsize}
    )
    ax.set_title(title, fontsize=fontsize)
    # set pie chart fontsize
    return fig, ax

def get_concept_grounding_info(outputs, query):
    assert 'execution_traces' in outputs and 'parsings' in outputs, 'Outputs should contain execution traces and parsings'
    assert len(outputs['execution_traces']) == 1, 'Only support batch size 1'
    grounding_tensors = {}
    grounding_tensors['query'] = outputs['parsings'][0]
    grounding_tensors['language'] = symbolic_to_natural_language(query)
    query_domain = create_domain_from_query([query])
    query_concept = list(query_domain.functions.keys())[3:]
    
    if len(query_concept) == 1:
        grounding_tensors['subject'] = {
            'name': query_concept[0],
            'tensor': outputs['execution_traces'][0][1][1].tensor.cpu().numpy()
        }

        return grounding_tensors
    elif len(query_concept) == 2:
        for i in range(len(outputs['execution_traces'][0])):
                if i == 1:
                    grounding_tensors['subject'] = {
                        'name': query_concept[0],
                        'tensor': outputs['execution_traces'][0][i][1].tensor.cpu().numpy()
                    }
                elif i == 4:
                    grounding_tensors['object'] = {
                        'name': query_concept[1],
                        'tensor': outputs['execution_traces'][0][i][1].tensor.cpu().numpy()
                    }
                else:
                    continue
        return grounding_tensors
                    
    elif len(query_concept) == 3:   
        # if grounding_tensors['language'].lower().startswith('what'):
        #     print(grounding_tensors['language'])
        #     print(query_concept)
        #     print(query)
        #     print(outputs['execution_traces'][0])
        #     print(len(outputs['execution_traces'][0]))
        #     for i in range(len(outputs['execution_traces'][0])):
        #         print(i)
        #         print(outputs['execution_traces'][0][i][0])
        #     exit()
        for i in range(len(outputs['execution_traces'][0])):
            if i == 1:
                # subject
                # assert '_Object' in outputs['execution_traces'][0][i][0].name, 'Expected _Object in execution trace'
                grounding_tensors['subject'] = {
                    'name': query_concept[0],
                    'tensor': outputs['execution_traces'][0][i][1].tensor.cpu().numpy()
                }
            elif i == 4:
                # object
                # assert '_Object' in outputs['execution_traces'][0][i][0].name, 'Expected _Object in execution trace'
                grounding_tensors['object'] = {
                    'name': query_concept[1],
                    'tensor': outputs['execution_traces'][0][i][1].tensor.cpu().numpy()
                }
            elif i == 6:
                # predicate
                # assert '_Object_Object' in outputs['execution_traces'][0][i][0].name, 'Expected _Object_Object in execution trace'
                grounding_tensors['predicate'] = {
                    'name': query_concept[2],
                    'tensor': outputs['execution_traces'][0][i][1].tensor.cpu().numpy()
                }
            else:
                continue
        return grounding_tensors
    
def plot_cdf(hop_stats, output_dir):
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(hop_stats))
    
    for idx, (hop, concepts) in enumerate(hop_stats.items()):
        # Get sorted frequencies for CDF
        freqs = np.sort(list(concepts.values()))
        cdf = np.arange(1, len(freqs) + 1) / len(freqs)
        
        plt.plot(freqs, cdf, 
                 label=f"{hop}-hop", 
                 color=colors[idx], 
                 linewidth=2)
    
    plt.xlabel("Frequency in Training Data", fontsize=12)
    plt.ylabel("Cumulative Proportion of Concepts", fontsize=12)
    plt.title("CDF of Concept Frequencies by Hop", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "cdf_concept_freq.png"), bbox_inches='tight')
    plt.close()

def plot_frequency_buckets(hop_stats, output_dir):
    # Define frequency bins (adjust as needed)
    buckets = [1, 5, 10, 20, 50, 100, 500, 1000, float("inf")]
    bin_edges = [0] + buckets  # Creates bins: [0-1), [1-5), [5-10), etc.
    labels = [
        "1", 
        "1-5", 
        "5-10", 
        "10-20", 
        "20-50", 
        "50-100",
        "100-500",
        "500-1000",
        "1000+"
    ]  # Ensure labels match bin edges
    
    # Create a color palette with one color per bucket
    num_buckets = len(labels)
    color_palette = sns.color_palette("viridis", num_buckets)
    
    plt.figure(figsize=(12, 6))
    
    for hop_idx, (hop, concepts) in enumerate(hop_stats.items()):
        freqs = list(concepts.values())
        counts, _ = np.histogram(freqs, bins=bin_edges)
        percentages = counts / len(freqs) * 100
        
        bottom = 0
        for bin_idx, percent in enumerate(percentages):
            plt.bar(
                hop, 
                percent, 
                bottom=bottom, 
                color=color_palette[bin_idx],  # Now matches buckets
                label=labels[bin_idx] if hop_idx == 0 else "",  # Avoid duplicate labels
                edgecolor='white'
            )
            bottom += percent
    
    plt.xlabel("Hop", fontsize=12)
    plt.ylabel("Percentage of Concepts", fontsize=12)
    plt.title("Distribution of Concepts by Frequency Bucket", fontsize=14)
    plt.legend(title="Frequency Range", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(output_dir, "frequency_buckets.png"), bbox_inches='tight')
    plt.close()

def plot_overlap_ratio(hop_stats, output_dir):
    overlap_ratios = {}
    for hop, concepts in hop_stats.items():
        total_concepts = len(concepts)
        overlapping = sum(1 for freq in concepts.values() if freq > 1)
        overlap_ratios[hop] = overlapping / total_concepts * 100
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(overlap_ratios.keys()), y=list(overlap_ratios.values()), 
                palette="Blues_d")
    plt.xlabel("Hop", fontsize=12)
    plt.ylabel("% of Concepts with Frequency > 1", fontsize=12)
    plt.title("Overlap Ratio (Concepts Appearing More Than Once)", fontsize=14)
    plt.savefig(os.path.join(output_dir, "overlap_ratio.png"), bbox_inches='tight')
    plt.close()

def plot_boxplot_swarm(hop_stats, output_dir):
    plt.figure(figsize=(12, 6))
    data = []
    for hop, concepts in hop_stats.items():
        for freq in concepts.values():
            data.append({"hop": hop, "frequency": freq})
    
    df = pd.DataFrame(data)
    
    sns.boxplot(x="hop", y="frequency", data=df, showfliers=False, palette="Pastel1")
    sns.swarmplot(x="hop", y="frequency", data=df, size=3, color=".2", alpha=0.5)
    
    plt.yscale("log")  # Use log scale if frequencies vary widely
    plt.xlabel("Hop", fontsize=12)
    plt.ylabel("Frequency (log scale)", fontsize=12)
    plt.title("Concept Frequency Distribution by Hop", fontsize=14)
    plt.savefig(os.path.join(output_dir, "boxplot_swarm.png"), bbox_inches='tight')
    plt.close()

def plot_example_images(model, test_dataset, output_dir):
    """
    Plot example images with their queries and predicted answers.
    
    Args:
        model: The trained model
        test_dataset: The test dataset
        output_dir: Directory to save the plots
    """
    logger.info(f"Plotting example images")
    model.eval()
    with torch.no_grad():
        for img_id, data in enumerate(test_dataset):
            data = {k: torch.Tensor(v).to('cuda').unsqueeze(0) if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v for k, v in data.items()}
            _data = deepcopy(data)
            for i in range(len(data['queries'])):
                try:
                    _data['queries'] = [data['queries'][i]]
                    _data['answers'] = [data['answers'][i]]
                    outputs = model(_data)
                    pred_answer = outputs['pred_answers'][0]
                    query = data['queries'][i]
                    answer = data['answers'][i]
                    
                    plt.figure()
                    plt.imshow(data['image'][0].cpu().numpy().transpose(1, 2, 0))
                    plt.title(f"Answer: {answer}\nPredicted answer: {pred_answer}")
                    plt.xlabel(query)
                    plt.savefig(os.path.join(output_dir, f"img_{img_id}_query_{i}.png"))
                    plt.close()
                except Exception as e:
                    logger.error(f"Error plotting image {img_id}, query {i}: {str(e)}")
                    continue

@torch.no_grad()
def evaluate_yes_no_results(model, test_dataset, output_dir, train_domain_concepts=None, model_name='ns'):
    """
    Evaluate model predictions on yes/no queries from test dataset, separated by hop.
    
    Args:
        model: The trained model
        test_dataset: The test dataset
        output_dir: Directory to save the plots
        train_domain_concepts: Set of concepts seen in training (optional). If provided,
                             queries with unseen concepts will be skipped.
        model_name: Name of the model
    Returns:
        dict: Statistics for each hop including answers, predictions, and success rates
    """
    # Initialize counters for all hops (1,2,3) and combined
    all_hops = [1, 2, 3]
    hop_stats = {hop: {'answers': [], 'pred_answers': [], 'num_fail': 0, 'total': 0} for hop in all_hops}
    hop_stats['all'] = {'answers': [], 'pred_answers': [], 'num_fail': 0, 'total': 0}
    
    # Initialize counters for attribute only and contains relation
    concept_type_stats = {
        'attr': {'answers': [], 'pred_answers': [], 'num_fail': 0, 'total': 0},
        'rel': {'answers': [], 'pred_answers': [], 'num_fail': 0, 'total': 0}
    }
    concept_grounding_info = {}
    total_fail = 0
    total_queries = 0
    model.eval()
    concept_grounding_info['network_names'] = list(model.scene_graph.network_dict.keys()) if model_name == 'ns' else None
    with torch.no_grad():
        for img_id, data in tqdm(enumerate(test_dataset)):
            data = {k: torch.Tensor(v).to('cuda').unsqueeze(0) if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v for k, v in data.items()}
            _data = deepcopy(data)
            if 'img_name' in data:
                img_name = data['img_name']
            else:
                img_name = f'{data["movie_name"]}_w{data["window_idx"]}'
            concept_grounding_info[img_name] = {}
            for i in range(len(data['queries'])):
                total_queries += 1
                try:
                    # Get the hop count for this query
                    query_domain = create_domain_from_query([data['queries'][i]])
                    query_concept = list(query_domain.functions.keys())[3:]
                    hop_count = len(query_concept)
                
                    # Get concept types
                    attr_concepts, rel_concepts, multi_rel_concepts = read_concepts_v2(query_domain)
                    
                    # Skip if hop not in valid range
                    if hop_count not in all_hops:
                        # print(f"Skipping query {i} for image {img_id} because hop count {hop_count} is not in valid range")
                        continue
                    
                    # Skip if contains untrained concepts
                    if train_domain_concepts is not None and any([c not in train_domain_concepts for c in query_concept]):
                        hop_stats[hop_count]['num_fail'] += 1
                        hop_stats['all']['num_fail'] += 1
                        total_fail += 1
                        # exit()
                        continue
                        
                    # skip if answer is not yes or no
                    if data['answers'][i] not in ['yes', 'no']:
                        continue
                    
                    _data['queries'] = [data['queries'][i]]
                    _data['answers'] = [data['answers'][i]]
                    outputs = model(_data)
                    pred_answer = outputs['pred_answers'][0]
                    answer = data['answers'][i]
                    if model_name == 'ns':
                        try:
                            concept_grounding_info[img_name][i] = get_concept_grounding_info(outputs, data['queries'][i])
                            concept_grounding_info[img_name][i]['pred_answer'] = pred_answer
                            concept_grounding_info[img_name][i]['answer'] = answer
                        except Exception as e:
                            logger.warning(f"Error getting concept grounding info for query {i} for image {img_id}: {str(e)}")
                            # continue
                    # Determine if this is a relation query
                    is_relation_query = any(c in rel_concepts or c in multi_rel_concepts for c in query_concept)
                    query_type = 'rel' if is_relation_query else 'attr'
                    
                    # Store results for this hop and combined
                    hop_stats[hop_count]['answers'].append(answer)
                    hop_stats[hop_count]['pred_answers'].append(pred_answer)
                    hop_stats[hop_count]['total'] += 1
                    
                    hop_stats['all']['answers'].append(answer)
                    hop_stats['all']['pred_answers'].append(pred_answer)
                    hop_stats['all']['total'] += 1
                    
                    # Store results for concept type
                    concept_type_stats[query_type]['answers'].append(answer)
                    concept_type_stats[query_type]['pred_answers'].append(pred_answer)
                    concept_type_stats[query_type]['total'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing query {i} for image {img_id}: {str(e)}")
                    continue
    
    # Create confusion matrices for each hop, overall, and concept types
    for hop in all_hops + ['all']:
        if hop_stats[hop]['total'] > 0:
            # Create confusion matrix
            cm = np.zeros((2, 2))
            for true, pred in zip(hop_stats[hop]['answers'], hop_stats[hop]['pred_answers']):
                true_idx = 1 if true == 'yes' else 0
                pred_idx = 1 if pred == 'yes' else 0
                cm[true_idx, pred_idx] += 1
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                        xticklabels=['no', 'yes'],
                        yticklabels=['no', 'yes'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for Hop {hop}' if hop != 'all' else 'Overall Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'hop_{hop}_cm.png'))
            plt.close()
            
            # Calculate metrics
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store metrics in stats
            hop_stats[hop]['metrics'] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'confusion_matrix': cm.tolist()
            }
    
    # Create confusion matrices for concept types
    for query_type in ['attr', 'rel']:
        if concept_type_stats[query_type]['total'] > 0:
            cm = np.zeros((2, 2))
            for true, pred in zip(concept_type_stats[query_type]['answers'], 
                                concept_type_stats[query_type]['pred_answers']):
                true_idx = 1 if true == 'yes' else 0
                pred_idx = 1 if pred == 'yes' else 0
                cm[true_idx, pred_idx] += 1
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                        xticklabels=['no', 'yes'],
                        yticklabels=['no', 'yes'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for {query_type.upper()} Queries')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'query_{query_type}_cm.png'))
            plt.close()
            
            # Calculate metrics
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store metrics in stats
            concept_type_stats[query_type]['metrics'] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'confusion_matrix': cm.tolist()
            }
    
    # Save all statistics
    stats_file = os.path.join(output_dir, "yes_no_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump({
            'hop_stats': hop_stats,
            'concept_type_stats': concept_type_stats
        }, f, indent=4)
    
    # Print final results in a more organized way
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY (yes/no queries)")
    print("="*50)
    
    # Print overall results
    print("\nOVERALL RESULTS (yes/no queries):")
    print(f"Total queries processed: {total_queries}")
    print(f"Failed queries: {total_fail}")
    print(f"Success rate: {(total_queries - total_fail) / total_queries:.4f}")
    
    # Print results for each hop
    print("\nRESULTS BY HOP (yes/no queries):")
    for hop in all_hops + ['all']:
        if hop_stats[hop]['total'] > 0:
            metrics = hop_stats[hop]['metrics']
            print(f"\nHop {hop}:")
            print(f"  Total queries: {hop_stats[hop]['total']}")
            print(f"  Failed queries: {hop_stats[hop]['num_fail']}")
            print(f"  Success rate: {(hop_stats[hop]['total'] - hop_stats[hop]['num_fail']) / hop_stats[hop]['total']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Print results by concept type
    print("\nRESULTS BY CONCEPT TYPE (yes/no queries):")
    for query_type in ['attr', 'rel']:
        if concept_type_stats[query_type]['total'] > 0:
            metrics = concept_type_stats[query_type]['metrics']
            print(f"\n{query_type.upper()} Queries:")
            print(f"  Total queries: {concept_type_stats[query_type]['total']}")
            print(f"  Failed queries: {concept_type_stats[query_type]['num_fail']}")
            print(f"  Success rate: {(concept_type_stats[query_type]['total'] - concept_type_stats[query_type]['num_fail']) / concept_type_stats[query_type]['total']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
    
    print("\n" + "="*50)
    # save concept grounding info to numpy file
    if model_name == 'ns':
        np.save(os.path.join(output_dir, "tf_concept_grounding_info.npy"), concept_grounding_info)
    return hop_stats

@torch.no_grad()
def evaluate_vocab_results(model, test_dataset, output_dir, vocab_type, train_domain_concepts=None, model_name='ns'):
    """
    Evaluate model predictions on vocabulary-based queries from test dataset.
    Skips yes/no queries and creates confusion matrix visualization.
    
    Args:
        model: The trained model
        test_dataset: The test dataset
        output_dir: Directory to save the plots and statistics
        vocab_type: Dictionary mapping queries to their vocabulary types, Dict[type: List[vocab]]
        train_domain_concepts: Set of concepts seen in training (optional). If provided,
                             queries with unseen concepts will be skipped.
        model_name: Name of the model
    Returns:
        dict: Statistics including answers, predictions, and success rates for each vocab type
    """
    # Initialize counters for each vocab type and overall
    stats = {
        'overall': {
            'answers': [],
            'pred_answers': [],
            'num_fail': 0,
            'total': 0,
            'queries': []
        }
    }
    
    # Initialize stats for each vocab type
    for vtype in set(vocab_type.keys()):
        stats[vtype] = {
            'answers': [],
            'pred_answers': [],
            'num_fail': 0,
            'total': 0,
            'queries': []
        }
    # reverse vocab_type
    vocab2type = {v: k for k, v in vocab_type.items() for v in v}
    model.eval()
    with torch.no_grad():
        for img_id, data in tqdm(enumerate(test_dataset)):
            data = {k: torch.Tensor(v).to('cuda').unsqueeze(0) if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v for k, v in data.items()}
            _data = deepcopy(data)
            for i in range(len(data['queries'])):
                try:
                    # Skip yes/no queries
                    if data['answers'][i] in ['yes', 'no']:
                        continue
                        
                    # Get concepts in query
                    query_domain = create_domain_from_query([data['queries'][i]])
                    query_concept = list(query_domain.functions.keys())[3:]
                    
                    # Skip if contains untrained concepts
                    if train_domain_concepts is not None and any([c not in train_domain_concepts for c in query_concept]):
                        stats['overall']['num_fail'] += 1
                        continue
                    
                    _data['queries'] = [data['queries'][i]]
                    _data['answers'] = [data['answers'][i]]
                    outputs = model(_data)
                    pred_answer = outputs['pred_answers'][0]
                    answer = data['answers'][i]
                    # Get vocab type for this query
                    query = data['queries'][i]
                    # check if answer is in vocab_type
                    vtype = vocab2type.get(answer, 'unknown')
                    # Store results for specific vocab type
                    if vtype in stats:
                        stats[vtype]['answers'].append(answer)
                        stats[vtype]['pred_answers'].append(pred_answer)
                        stats[vtype]['queries'].append(query)
                        stats[vtype]['total'] += 1
                    
                    # Store results for overall
                    stats['overall']['answers'].append(answer)
                    stats['overall']['pred_answers'].append(pred_answer)
                    stats['overall']['queries'].append(query)
                    stats['overall']['total'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing query {i} for image {img_id}: {str(e)}")
                    continue
    # Create confusion matrices and calculate metrics for each vocab type
    for vtype in stats:
        if stats[vtype]['total'] > 0:
            # Get unique answers and create mapping
            unique_answers = sorted(list(set(stats[vtype]['answers'])))
            answer_to_idx = {ans: idx for idx, ans in enumerate(unique_answers)}
            
            # Create confusion matrix
            cm = np.zeros((len(unique_answers), len(unique_answers)))
            for true, pred in zip(stats[vtype]['answers'], stats[vtype]['pred_answers']):
                if true in answer_to_idx and pred in answer_to_idx:
                    cm[answer_to_idx[true], answer_to_idx[pred]] += 1
            
            # Filter out rows and columns with all zeros (both pred and true have 0 samples)
            row_sums = cm.sum(axis=1)
            col_sums = cm.sum(axis=0)
            # Keep indices where either row or column has non-zero values
            keep_indices = np.where((row_sums > 0) | (col_sums > 0))[0]
            
            if len(keep_indices) > 0:
                # Filter confusion matrix and labels
                cm_filtered = cm[np.ix_(keep_indices, keep_indices)]
                unique_answers_filtered = [unique_answers[i] for i in keep_indices]
            else:
                # If all are zero, keep original (shouldn't happen if total > 0)
                cm_filtered = cm
                unique_answers_filtered = unique_answers
            
            # Plot confusion matrix (counts)
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm_filtered, annot=True, fmt='g', cmap='Blues',
                        xticklabels=unique_answers_filtered,
                        yticklabels=unique_answers_filtered)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for {vtype.upper()} Vocabulary Queries')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'vocab_confusion_matrix_{vtype}.png'))
            plt.close()
            
            # Create probability confusion matrix (normalized by row, each row sums to 1)
            cm_prob = cm_filtered.copy().astype(float)
            row_sums_filtered = cm_prob.sum(axis=1, keepdims=True)
            # Normalize by row, handling division by zero
            cm_prob = np.divide(cm_prob, row_sums_filtered, 
                               out=np.zeros_like(cm_prob), 
                               where=row_sums_filtered!=0)
            
            # Plot probability confusion matrix
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm_prob*100.0, annot=True, fmt='.1f', cmap='Blues',
                        xticklabels=unique_answers_filtered,
                        yticklabels=unique_answers_filtered,
                        cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Probability Confusion Matrix for {vtype.upper()} Vocabulary Queries')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'vocab_confusion_matrix_{vtype}_prob.png'))
            plt.close()
            
            # Calculate accuracy
            correct = sum(1 for true, pred in zip(stats[vtype]['answers'], stats[vtype]['pred_answers']) if true == pred)
            accuracy = correct / stats[vtype]['total'] if stats[vtype]['total'] > 0 else 0
            
            # Store metrics
            stats[vtype]['accuracy'] = accuracy
            stats[vtype]['unique_answers'] = unique_answers
            stats[vtype]['confusion_matrix'] = cm.tolist()
    
    # Save statistics
    stats_file = os.path.join(output_dir, "vocab_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Print results
    print("\n" + "="*50)
    print("VOCABULARY QUERY RESULTS")
    print("="*50)
    
    # Print overall results
    print("\nOVERALL RESULTS:")
    print(f"Total queries processed: {stats['overall']['total']}")
    print(f"Failed queries: {stats['overall']['num_fail']}")
    print(f"Accuracy: {stats['overall']['accuracy']:.4f}")
    print(f"Number of unique answers: {len(stats['overall']['unique_answers'])}")
    
    # Print results for each vocab type
    for vtype in stats:
        if vtype != 'overall' and stats[vtype]['total'] > 0:
            print(f"\n{vtype.upper()} RESULTS:")
            print(f"Total queries processed: {stats[vtype]['total']}")
            print(f"Failed queries: {stats[vtype]['num_fail']}")
            print(f"Accuracy: {stats[vtype]['accuracy']:.4f}")
            print(f"Number of unique answers: {len(stats[vtype]['unique_answers'])}")
    
    print("\n" + "="*50)
    
    return stats

@torch.no_grad()
def evaluate_systematicity_results(model, test_dataset, train_programs, output_dir, train_domain_concepts=None, model_name='ns'):
    """
    Evaluate model predictions on test queries and compute systematicity accuracy.
    Classifies queries into argument swapping, predicate transfer, and role systematicity.
    
    Args:
        model: The trained model
        test_dataset: The test dataset
        train_programs: List of training query strings
        output_dir: Directory to save the results
        train_domain_concepts: Set of concepts seen in training (optional). If provided,
                             queries with unseen concepts will be skipped.
        model_name: Name of the model
    Returns:
        dict: Systematicity accuracy results for each category
    """
    # Collect queries, answers, and predictions
    test_queries = []
    test_answers = []
    test_pred_answers = []
    
    model.eval()
    with torch.no_grad():
        for img_id, data in tqdm(enumerate(test_dataset), desc="Collecting systematicity data"):
            data = {k: torch.Tensor(v).to('cuda').unsqueeze(0) if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v for k, v in data.items()}
            _data = deepcopy(data)
            for i in range(len(data['queries'])):
                try:
                    # Only process yes/no queries for systematicity analysis
                    if data['answers'][i] not in ['yes', 'no']:
                        continue
                    
                    # Get concepts in query
                    query_domain = create_domain_from_query([data['queries'][i]])
                    query_concept = list(query_domain.functions.keys())[3:]
                    
                    # Skip if contains untrained concepts
                    if train_domain_concepts is not None and any([c not in train_domain_concepts for c in query_concept]):
                        continue
                    
                    _data['queries'] = [data['queries'][i]]
                    _data['answers'] = [data['answers'][i]]
                    outputs = model(_data)
                    pred_answer = outputs['pred_answers'][0]
                    answer = data['answers'][i]
                    
                    # Store for systematicity analysis
                    test_queries.append(data['queries'][i])
                    test_answers.append(answer)
                    test_pred_answers.append(pred_answer)
                    
                except Exception as e:
                    logger.error(f"Error processing query {i} for image {img_id}: {str(e)}")
                    continue
    
    # Compute systematicity accuracy
    if len(test_queries) > 0 and len(train_programs) > 0:
        logger.info(f"Computing systematicity accuracy for {len(test_queries)} queries...")
        results = compute_systematicity_accuracy(
            test_queries=test_queries,
            test_answers=test_answers,
            test_pred_answers=test_pred_answers,
            train_queries=train_programs
        )
        
        # Save results
        results_file = os.path.join(output_dir, "systematicity_results.json")
        # Convert to JSON-serializable format
        results_serializable = {}
        for category, data in results.items():
            results_serializable[category] = {
                'accuracy': float(data['accuracy']) if data['accuracy'] is not None else None,
                'total': int(data['total']),
                'correct': int(data['correct']),
                'num_queries': len(data['queries'])
            }
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=4)
        
        # Print and save report
        report_file = os.path.join(output_dir, "systematicity_report.txt")
        report_systematicity_accuracy(results, output_file=report_file)
        
        logger.info(f"Systematicity results saved to {results_file}")
        logger.info(f"Systematicity report saved to {report_file}")
        
        return results
    else:
        logger.warning("No queries collected for systematicity analysis or no train programs provided")
        return None

def read_evaluation_stats(output_dir):
    """
    Read evaluation statistics from JSON files in the output directory.
    
    Args:
        output_dir: Directory containing the evaluation statistics JSON files
        
    Returns:
        tuple: (hop_stats, vocab_stats, systematicity_stats) containing the statistics
    """
    import json
    import os

    # Read yes/no statistics
    yes_no_stats_file = os.path.join(output_dir, "yes_no_statistics.json")
    if os.path.exists(yes_no_stats_file):
        with open(yes_no_stats_file, 'r') as f:
            yes_no_stats = json.load(f)
            hop_stats = yes_no_stats['hop_stats']
    else:
        hop_stats = None
    
    # Read vocabulary statistics
    vocab_stats_file = os.path.join(output_dir, "vocab_statistics.json")
    if os.path.exists(vocab_stats_file):
        with open(vocab_stats_file, 'r') as f:
            vocab_stats = json.load(f)
    else:
        vocab_stats = None
    
    # Read systematicity statistics
    systematicity_stats_file = os.path.join(output_dir, "systematicity_results.json")
    if os.path.exists(systematicity_stats_file):
        with open(systematicity_stats_file, 'r') as f:
            systematicity_stats = json.load(f)
    else:
        systematicity_stats = None
    
    return hop_stats, vocab_stats, systematicity_stats    

def log_evaluation_result(output_dir):
    """
    Read and log evaluation results from the output directory.
    
    Args:
        output_dir: Directory containing the evaluation statistics
    """
    
    # Set up logging
    log_file = os.path.join(output_dir, "evaluation_results.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Read statistics
    hop_stats, vocab_stats, systematicity_stats = read_evaluation_stats(output_dir)
    # Log hop statistics
    if hop_stats:
        logging.info("\n" + "="*50)
        logging.info("HOP STATISTICS (yes/no queries)")
        logging.info("="*50)
        
        for hop in [1, 2, 3, 'all']:
            if hop in hop_stats and hop_stats[hop]['total'] > 0:
                metrics = hop_stats[hop]['metrics']
                logging.info(f"\nHop {hop}:")
                logging.info(f"  Total queries: {hop_stats[hop]['total']}")
                logging.info(f"  Failed queries: {hop_stats[hop]['num_fail']}")
                logging.info(f"  Success rate: {(hop_stats[hop]['total'] - hop_stats[hop]['num_fail']) / hop_stats[hop]['total']:.4f}")
                logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logging.info(f"  Precision: {metrics['precision']:.4f}")
                logging.info(f"  Recall: {metrics['recall']:.4f}")
                logging.info(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Log vocabulary statistics
    if vocab_stats:
        logging.info("\n" + "="*50)
        logging.info("VOCABULARY QUERY STATISTICS")
        logging.info("="*50)
        
        # Log overall vocabulary results
        if 'overall' in vocab_stats and vocab_stats['overall']['total'] > 0:
            logging.info("\nOVERALL VOCABULARY RESULTS:")
            logging.info(f"Total queries processed: {vocab_stats['overall']['total']}")
            logging.info(f"Failed queries: {vocab_stats['overall']['num_fail']}")
            logging.info(f"Accuracy: {vocab_stats['overall']['accuracy']:.4f}")
            logging.info(f"Number of unique answers: {len(vocab_stats['overall']['unique_answers'])}")
        
        # Log results for each vocabulary type
        for vtype in vocab_stats:
            if vtype != 'overall' and vocab_stats[vtype]['total'] > 0:
                logging.info(f"\n{vtype.upper()} VOCABULARY RESULTS:")
                logging.info(f"Total queries processed: {vocab_stats[vtype]['total']}")
                logging.info(f"Failed queries: {vocab_stats[vtype]['num_fail']}")
                logging.info(f"Accuracy: {vocab_stats[vtype]['accuracy']:.4f}")
                logging.info(f"Number of unique answers: {len(vocab_stats[vtype]['unique_answers'])}")
    
    # Log systematicity statistics
    if systematicity_stats:
        logging.info("\n" + "="*50)
        logging.info("SYSTEMATICITY STATISTICS")
        logging.info("="*50)
        
        category_names = {
            'argument_swapping': 'Argument Swapping',
            'predicate_transfer': 'Predicate Transfer',
            'role_systematicity': 'Role Systematicity',
            'none': 'Other Queries'
        }
        
        categories = ['argument_swapping', 'predicate_transfer', 'role_systematicity', 'none']
        for category in categories:
            if category in systematicity_stats and systematicity_stats[category]['total'] > 0:
                data = systematicity_stats[category]
                name = category_names.get(category, category)
                logging.info(f"\n{name}:")
                if data['accuracy'] is not None:
                    logging.info(f"  Accuracy: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")
                else:
                    logging.info(f"  Accuracy: N/A")
                logging.info(f"  Total queries: {data['total']}")
                logging.info(f"  Correct predictions: {data['correct']}")
    
    # Log overall results
    logging.info("\n" + "="*50)
    logging.info("OVERALL RESULTS")
    logging.info("="*50)
    
    if hop_stats is None:
        logging.warning("hop_stats is None!")
    if vocab_stats is None:
        logging.warning("vocab_stats is None!")
    if systematicity_stats is None:
        logging.warning("systematicity_stats is None!")
    
    # Calculate and log combined accuracy
    if hop_stats and 'all' in hop_stats and vocab_stats and 'overall' in vocab_stats:
        overall_hop = hop_stats['all']
        overall_vocab = vocab_stats['overall']
        
        total_queries = overall_hop['total'] + overall_vocab['total']
        if total_queries > 0:
            combined_accuracy = (
                (overall_hop['metrics']['accuracy'] * overall_hop['total'] + 
                 overall_vocab['accuracy'] * overall_vocab['total']) / 
                total_queries
            )
            
            logging.info("\nCOMBINED RESULTS (Yes/No + Vocabulary):")
            logging.info(f"Total queries: {total_queries}")
            logging.info(f"Combined accuracy: {combined_accuracy:.4f}")
            logging.info(f"Yes/No queries: {overall_hop['total']} ({overall_hop['metrics']['accuracy']:.4f})")
            logging.info(f"Vocabulary queries: {overall_vocab['total']} ({overall_vocab['accuracy']:.4f})")
    
    logging.info("\n" + "="*50)

def log_evaluation_results(output_dirs: List[str]):
    """
    Read and log evaluation results from multiple output directories.
    
    Args:
        output_dirs: List of directories containing the evaluation statistics
    """
    # Initialize aggregated statistics
    aggregated_hop_stats = {}
    aggregated_vocab_stats = {}
    aggregated_systematicity_stats = {}
    
    # Store individual subject values for std calculation
    subject_hop_metrics = {}  # {hop: {metric: [values]}}
    subject_vocab_accuracy = {}  # {vtype: [values]}
    subject_systematicity_accuracy = {}  # {category: [values]}
    assert len(output_dirs) > 0, "No output directories provided"
    # Process each output directory
    for output_dir in output_dirs:
        subject = output_dir.split('sub-')[1].split('/')[0]
        
        # Set up logging for this subject
        log_file = os.path.join(output_dir, "evaluation_results.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Read statistics for this subject
        hop_stats, vocab_stats, systematicity_stats = read_evaluation_stats(output_dir)
        # Log subject-specific results
        logging.info(f"\nProcessing results for subject: {subject}")
        
        # Log hop statistics
        if hop_stats:
            logging.info("\n" + "="*50)
            logging.info("HOP STATISTICS (yes/no queries)")
            logging.info("="*50)
            
            for hop in [1, 2, 3, 'all']:
                if hop in hop_stats and hop_stats[hop]['total'] > 0:
                    metrics = hop_stats[hop]['metrics']
                    logging.info(f"\nHop {hop}:")
                    logging.info(f"  Total queries: {hop_stats[hop]['total']}")
                    logging.info(f"  Failed queries: {hop_stats[hop]['num_fail']}")
                    logging.info(f"  Success rate: {(hop_stats[hop]['total'] - hop_stats[hop]['num_fail']) / hop_stats[hop]['total']:.4f}")
                    logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                    logging.info(f"  Precision: {metrics['precision']:.4f}")
                    logging.info(f"  Recall: {metrics['recall']:.4f}")
                    logging.info(f"  F1 Score: {metrics['f1']:.4f}")
                    
                    # Store individual subject values for std calculation
                    if hop not in subject_hop_metrics:
                        subject_hop_metrics[hop] = {
                            'accuracy': [],
                            'precision': [],
                            'recall': [],
                            'f1': []
                        }
                    for metric in ['accuracy', 'precision', 'recall', 'f1']:
                        if 'metrics' in hop_stats[hop]:
                            subject_hop_metrics[hop][metric].append(hop_stats[hop]['metrics'][metric])
        
        # Log vocabulary statistics
        if vocab_stats:
            logging.info("\n" + "="*50)
            logging.info("VOCABULARY QUERY STATISTICS")
            logging.info("="*50)

            # Log results for each vocabulary type
            for vtype in vocab_stats:
                if vocab_stats[vtype]['total'] > 0:
                    logging.info(f"\n{vtype.upper()} VOCABULARY RESULTS:")
                    logging.info(f"Total queries processed: {vocab_stats[vtype]['total']}")
                    logging.info(f"Failed queries: {vocab_stats[vtype]['num_fail']}")
                    logging.info(f"Accuracy: {vocab_stats[vtype]['accuracy']:.4f}")
                    logging.info(f"Number of unique answers: {len(vocab_stats[vtype]['unique_answers'])}")
                    
                    # Store individual subject values for std calculation
                    if vtype not in subject_vocab_accuracy:
                        subject_vocab_accuracy[vtype] = []
                    subject_vocab_accuracy[vtype].append(vocab_stats[vtype]['accuracy'])
        
        # Log systematicity statistics
        if systematicity_stats:
            logging.info("\n" + "="*50)
            logging.info("SYSTEMATICITY STATISTICS")
            logging.info("="*50)
            
            category_names = {
                'argument_swapping': 'Argument Swapping',
                'predicate_transfer': 'Predicate Transfer',
                'role_systematicity': 'Role Systematicity',
                'none': 'Other Queries'
            }
            
            categories = ['argument_swapping', 'predicate_transfer', 'role_systematicity', 'none']
            for category in categories:
                if category in systematicity_stats and systematicity_stats[category]['total'] > 0:
                    data = systematicity_stats[category]
                    name = category_names.get(category, category)
                    logging.info(f"\n{name}:")
                    if data['accuracy'] is not None:
                        logging.info(f"  Accuracy: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")
                    else:
                        logging.info(f"  Accuracy: N/A")
                    logging.info(f"  Total queries: {data['total']}")
                    logging.info(f"  Correct predictions: {data['correct']}")
                    
                    # Store individual subject values for std calculation
                    if category not in subject_systematicity_accuracy:
                        subject_systematicity_accuracy[category] = []
                    if data['accuracy'] is not None:
                        subject_systematicity_accuracy[category].append(data['accuracy'])

        # Aggregate statistics
        if hop_stats:
            for hop in hop_stats:
                if hop not in aggregated_hop_stats:
                    aggregated_hop_stats[hop] = {
                        'total': 0,
                        'num_fail': 0,
                        'metrics': {
                            'accuracy': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'f1': 0.0
                        }
                    }
                aggregated_hop_stats[hop]['total'] += hop_stats[hop]['total']
                aggregated_hop_stats[hop]['num_fail'] += hop_stats[hop]['num_fail']
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    if 'metrics' in hop_stats[hop]:
                        aggregated_hop_stats[hop]['metrics'][metric] += hop_stats[hop]['metrics'][metric]
        
        if vocab_stats:
            for vtype in vocab_stats:
                if vocab_stats[vtype]['total'] <= 0:
                    continue
                if vtype not in aggregated_vocab_stats:
                    aggregated_vocab_stats[vtype] = {
                        'total': 0,
                        'num_fail': 0,
                        'accuracy': 0.0
                    }
                aggregated_vocab_stats[vtype]['total'] += vocab_stats[vtype]['total']
                aggregated_vocab_stats[vtype]['num_fail'] += vocab_stats[vtype]['num_fail']
                aggregated_vocab_stats[vtype]['accuracy'] += vocab_stats[vtype]['accuracy']
        
        # Aggregate systematicity statistics
        if systematicity_stats:
            categories = ['argument_swapping', 'predicate_transfer', 'role_systematicity', 'none']
            for category in categories:
                if category in systematicity_stats and systematicity_stats[category]['total'] > 0:
                    if category not in aggregated_systematicity_stats:
                        aggregated_systematicity_stats[category] = {
                            'total': 0,
                            'correct': 0,
                            'accuracy': 0.0
                        }
                    aggregated_systematicity_stats[category]['total'] += systematicity_stats[category]['total']
                    aggregated_systematicity_stats[category]['correct'] += systematicity_stats[category]['correct']
                    if systematicity_stats[category]['accuracy'] is not None:
                        aggregated_systematicity_stats[category]['accuracy'] += systematicity_stats[category]['accuracy']
    
    # Calculate averages and standard deviations for aggregated statistics
    num_subjects = len(output_dirs)
    for hop in aggregated_hop_stats:
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if hop in subject_hop_metrics and metric in subject_hop_metrics[hop]:
                values = subject_hop_metrics[hop][metric]
                if values:
                    mean = sum(values) / len(values)
                    std = np.std(values) if len(values) > 1 else 0
                    aggregated_hop_stats[hop]['metrics'][metric] = {
                        'mean': mean,
                        'std': std
                    }
    for vtype in aggregated_vocab_stats:
        if vtype in subject_vocab_accuracy and subject_vocab_accuracy[vtype]:
            values = subject_vocab_accuracy[vtype]
            mean = sum(values) / len(values)
            std = np.std(values) if len(values) > 1 else 0
            aggregated_vocab_stats[vtype]['accuracy'] = {
                'mean': mean,
                'std': std
            }
        else:
            print(f"Skipping {vtype} because {subject_vocab_accuracy.keys()}")
            # print(subject_vocab_accuracy.keys())
    
    # Calculate averages and standard deviations for systematicity statistics
    for category in aggregated_systematicity_stats:
        if category in subject_systematicity_accuracy and subject_systematicity_accuracy[category]:
            values = subject_systematicity_accuracy[category]
            mean = sum(values) / len(values)
            std = np.std(values) if len(values) > 1 else 0
            aggregated_systematicity_stats[category]['accuracy'] = {
                'mean': mean,
                'std': std
            }
        else:
            # If no subject values, calculate accuracy from aggregated total/correct
            data = aggregated_systematicity_stats[category]
            if data['total'] > 0:
                accuracy = data['correct'] / data['total']
                aggregated_systematicity_stats[category]['accuracy'] = {
                    'mean': accuracy,
                    'std': 0.0
                }
            else:
                aggregated_systematicity_stats[category]['accuracy'] = None

    # Log aggregated results
    aggregated_log_file = os.path.join(os.path.dirname(output_dirs[0]), "aggregated_evaluation_results.log")
    logging.basicConfig(
        filename=aggregated_log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info("\n" + "="*50)
    logging.info("AGGREGATED RESULTS ACROSS SUBJECTS")
    logging.info("="*50)
    
    # Log aggregated hop statistics
    if aggregated_hop_stats:
        logging.info("\nAGGREGATED HOP STATISTICS:")
        for hop in [1, 2, 3, 'all']:
            if hop in aggregated_hop_stats:
                metrics = aggregated_hop_stats[hop]['metrics']
                logging.info(f"\nHop {hop}:")
                logging.info(f"  Total queries: {aggregated_hop_stats[hop]['total']}")
                logging.info(f"  Failed queries: {aggregated_hop_stats[hop]['num_fail']}")
                logging.info(f"  Success rate: {(aggregated_hop_stats[hop]['total'] - aggregated_hop_stats[hop]['num_fail']) / aggregated_hop_stats[hop]['total']:.4f}")
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    if metric in metrics and isinstance(metrics[metric], dict):
                        logging.info(f"  {metric.capitalize()}: {metrics[metric]['mean']:.4f} ± {metrics[metric]['std']:.4f}")
    
    # Log aggregated vocabulary statistics
    if aggregated_vocab_stats:
        logging.info("\nAGGREGATED VOCABULARY STATISTICS:")
        if 'overall' in aggregated_vocab_stats:
            logging.info("\nOVERALL VOCABULARY RESULTS:")
            logging.info(f"Total queries processed: {aggregated_vocab_stats['overall']['total']}")
            logging.info(f"Failed queries: {aggregated_vocab_stats['overall']['num_fail']}")
            if isinstance(aggregated_vocab_stats['overall']['accuracy'], dict):
                logging.info(f"Accuracy: {aggregated_vocab_stats['overall']['accuracy']['mean']:.4f} ± {aggregated_vocab_stats['overall']['accuracy']['std']:.4f}")
            else:
                logging.info(f"Accuracy: {aggregated_vocab_stats['overall']['accuracy']:.4f}")
        
        for vtype in aggregated_vocab_stats:
            if vtype != 'overall':
                logging.info(f"\n{vtype.upper()} VOCABULARY RESULTS:")
                logging.info(f"Total queries processed: {aggregated_vocab_stats[vtype]['total']}")
                logging.info(f"Failed queries: {aggregated_vocab_stats[vtype]['num_fail']}")
                if isinstance(aggregated_vocab_stats[vtype]['accuracy'], dict):
                    logging.info(f"Accuracy: {aggregated_vocab_stats[vtype]['accuracy']['mean']:.4f} ± {aggregated_vocab_stats[vtype]['accuracy']['std']:.4f}")
                else:
                    logging.info(f"Accuracy: {aggregated_vocab_stats[vtype]['accuracy']:.4f}")
    
    # Log aggregated systematicity statistics
    if aggregated_systematicity_stats:
        logging.info("\nAGGREGATED SYSTEMATICITY STATISTICS:")
        category_names = {
            'argument_swapping': 'Argument Swapping',
            'predicate_transfer': 'Predicate Transfer',
            'role_systematicity': 'Role Systematicity',
            'none': 'Other Queries'
        }
        
        categories = ['argument_swapping', 'predicate_transfer', 'role_systematicity', 'none']
        for category in categories:
            if category in aggregated_systematicity_stats and aggregated_systematicity_stats[category]['total'] > 0:
                data = aggregated_systematicity_stats[category]
                name = category_names.get(category, category)
                logging.info(f"\n{name}:")
                if isinstance(data['accuracy'], dict):
                    logging.info(f"  Accuracy: {data['accuracy']['mean']:.4f} ± {data['accuracy']['std']:.4f}")
                elif data['accuracy'] is not None:
                    logging.info(f"  Accuracy: {data['accuracy']:.4f}")
                else:
                    logging.info(f"  Accuracy: N/A")
                logging.info(f"  Total queries: {data['total']}")
                logging.info(f"  Correct predictions: {data['correct']}")
    
    # Calculate and log combined overall accuracy
    if aggregated_hop_stats and 'all' in aggregated_hop_stats and aggregated_vocab_stats and 'overall' in aggregated_vocab_stats:
        overall_hop = aggregated_hop_stats['all']
        overall_vocab = aggregated_vocab_stats['overall']
        
        total_queries = overall_hop['total'] + overall_vocab['total']
        if total_queries > 0:
            # Get accuracy values
            hop_accuracy = overall_hop['metrics']['accuracy']['mean'] if isinstance(overall_hop['metrics']['accuracy'], dict) else overall_hop['metrics']['accuracy']
            vocab_accuracy = overall_vocab['accuracy']['mean'] if isinstance(overall_vocab['accuracy'], dict) else overall_vocab['accuracy']
            
            # Calculate weights
            hop_weight = overall_hop['total'] / total_queries
            vocab_weight = overall_vocab['total'] / total_queries

            # Calculate combined accuracy
            combined_accuracy = hop_accuracy * hop_weight + vocab_accuracy * vocab_weight
            
            # Calculate combined standard deviation if available
            if isinstance(overall_hop['metrics']['accuracy'], dict) and isinstance(overall_vocab['accuracy'], dict):
                hop_std = overall_hop['metrics']['accuracy']['std']
                vocab_std = overall_vocab['accuracy']['std']
                combined_std = np.sqrt(
                    (hop_std**2 * hop_weight**2) + 
                    (vocab_std**2 * vocab_weight**2)
                )
                std_str = f" ± {combined_std:.4f}"
            else:
                std_str = ""
            
            # Format hop accuracy string
            if isinstance(overall_hop['metrics']['accuracy'], dict):
                hop_acc_str = f"{hop_accuracy:.4f} ± {overall_hop['metrics']['accuracy']['std']:.4f}"
            else:
                hop_acc_str = f"{hop_accuracy:.4f}"
            
            # Format vocab accuracy string
            if isinstance(overall_vocab['accuracy'], dict):
                vocab_acc_str = f"{vocab_accuracy:.4f} ± {overall_vocab['accuracy']['std']:.4f}"
            else:
                vocab_acc_str = f"{vocab_accuracy:.4f}"
            
            logging.info("\nCOMBINED RESULTS (Yes/No + Vocabulary):")
            logging.info(f"Total queries: {total_queries}")
            logging.info(f"Combined accuracy: {combined_accuracy:.4f}{std_str}")
            logging.info(f"Yes/No queries: {overall_hop['total']} ({hop_acc_str})")
            logging.info(f"Vocabulary queries: {overall_vocab['total']} ({vocab_acc_str})")
    
    logging.info("\n" + "="*50)

