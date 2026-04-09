import json
import os

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from tqdm import tqdm

from loader.wave import preprocess_sample
from models.left.domain import (create_default_parser,
                                create_domain_from_query, read_concepts_v2,
                                read_description_categories)
from utils.dataset_utils import (generate_describe_queries, generate_queries,
                                 get_tar_file_list, stats_concepts)
from utils.utils import get_args, set_seed


def main(args):

    scene_graph_path = 'data/bold5000_scene_graph.csv'
    pred_cls_path = 'data/bold5000_pred_cls.json'
    # load pred_cls
    with open(pred_cls_path, 'r') as f:
        pred_cls = json.load(f)
    pred_2_cls = {}
    for k, v in pred_cls.items():
        for pred in v:
            pred_2_cls[pred] = k
    scene_graph_df = pd.read_csv(scene_graph_path)
    tar_file_list = get_tar_file_list(
        tar_dir=args.wave_dir,
        subject_list=[args.subject],
        split_list=['train', 'test']
    )
    wave_dataset = wds.WebDataset(tar_file_list).decode("pil").map(preprocess_sample)
    save_dir = 'data/fqa'
    fail_samples = []
    num_pos_queries = 0
    num_neg_queries = 0
    all_queries = {
        'train': {
            'pos': [],
            'neg': []
        },
        'test': {
            'pos': [],
            'neg': []
        }
    }
        
    for i, sample in enumerate(tqdm(wave_dataset)):
        config = sample['config.json']
        is_test = config['rep']
        subj = sample['__key__'].split('sub-')[1].split('_')[0]
        img_name = sample['img_name']
        # find scene_graph which has the same image name
        scene_graph = scene_graph_df[scene_graph_df['img_name'] == img_name]['scene_graph'].values[0]
        # remove characters like * and #
        scene_graph = scene_graph.replace('*', '')
        scene_graph = scene_graph.replace('#', '')
        scene_graph = scene_graph.replace('-', '')
        # remove empty lines
        scene_graph = '\n'.join([line for line in scene_graph.split('\n') if line.strip() != ''])
        scene_graph = scene_graph.lower()
        
        try:
            pos_scene_graph = scene_graph.split('negative samples')[0].strip()
            neg_scene_graph = scene_graph.split('negative samples')[1].strip()
        except Exception as e:
            fail_samples.append(img_name)
            print(f'fail to parse {img_name} {e}')
            print(scene_graph)
            continue
        _pos_queries = generate_queries(pos_scene_graph)
        # remove failed queries
        pos_queries = []
        for query in _pos_queries:
            domain = create_domain_from_query([query])
            parser = create_default_parser(domain)
            if len(domain.functions.keys()) > 3:
                try:
                    x = parser.parse_expression(query)
                    pos_queries.append(query)
                except Exception as e:
                    print(f'fail to parse {query} {e}')
                    fail_samples.append(img_name)
        _neg_queries = generate_queries(neg_scene_graph)
        neg_queries = []
        for query in _neg_queries:
            domain = create_domain_from_query([query])
            parser = create_default_parser(domain)
            if len(domain.functions.keys()) > 3:
                try:
                    x = parser.parse_expression(query)
                    neg_queries.append(query)
                except Exception as e:
                    print(f'fail to parse {query} {e}')
                    fail_samples.append(img_name)
        all_queries['train']['pos'].extend(pos_queries) if not is_test else all_queries['test']['pos'].extend(pos_queries)
        all_queries['train']['neg'].extend(neg_queries) if not is_test else all_queries['test']['neg'].extend(neg_queries)

        
        num_pos_queries += len(pos_queries)
        num_neg_queries += len(neg_queries)

    print(f"Total train positive queries: {len(all_queries['train']['pos'])}, total train negative queries: {len(all_queries['train']['neg'])}, Total train queries: {len(all_queries['train']['pos']) + len(all_queries['train']['neg'])}")
    print(f"Total test positive queries: {len(all_queries['test']['pos'])}, total test negative queries: {len(all_queries['test']['neg'])}, Total test queries: {len(all_queries['test']['pos']) + len(all_queries['test']['neg'])}")
    train_queries = all_queries['train']['pos'] + all_queries['train']['neg']
    test_queries = all_queries['test']['pos'] + all_queries['test']['neg']
    train_domain = create_domain_from_query(train_queries)
    test_domain = create_domain_from_query(test_queries)
    pos_train_domain = create_domain_from_query(all_queries['train']['pos'])
    pos_train_domain_concepts = set(pos_train_domain.functions.keys())
    _, pos_train_relation_concepts, _ = read_concepts_v2(domain=pos_train_domain)
    pos_test_domain = create_domain_from_query(all_queries['test']['pos'])
    pos_test_domain_concepts = set(pos_test_domain.functions.keys())
    # check how many test domain functions are not in train domain
    test_domain_concepts = set(test_domain.functions.keys())
    train_domain_concepts = set(train_domain.functions.keys())
    print(f"Total train domain concepts: {len(train_domain_concepts)}, total test domain concepts: {len(test_domain_concepts)}")
    untrain_concepts = test_domain_concepts - train_domain_concepts
    # print(f"Test domain functions not in train domain: {untrain_concepts}")
    print(f"Number of test domain functions not in train domain: {len(untrain_concepts)}")
    #########################################################
    # stats train domain queries
    train_domain_queries = stats_concepts(train_queries)
    pos_train_domain_queries = stats_concepts(all_queries['train']['pos'])
    test_concepts_count = {}
    for concept in test_domain_concepts:
        if concept in train_domain_queries:
            test_concepts_count[concept] = train_domain_queries[concept]
        else:
            test_concepts_count[concept] = 0
    # calculate how many test concepts are only appear 0~5 times
    test_concepts_count = {k: v for k, v in test_concepts_count.items() if v >= 0 and v <= 5}
    print(f"Test concepts only appear 0~5 times: {len(test_concepts_count)}")
    # select test queries that only appear 0~5 times
    uncommon_test_queries = []
    for query in test_queries:
        domain = create_domain_from_query([query])
        concepts = list(domain.functions.keys())[3:]
        if any(concept in test_concepts_count for concept in concepts):
            uncommon_test_queries.append(query)
    print(f"Number of uncommon test queries: {len(uncommon_test_queries)}")
    
    # get clean test concepts, make sure true/false queries are clean now
    clean_test_concepts = []
    for concept in test_domain_concepts:
        if concept not in test_concepts_count:
            clean_test_concepts.append(concept)
    print(f"Number of clean test concepts: {len(clean_test_concepts)}")

    # get clean positve test relational concepts
    pos_test_relational_concepts = []
    for concept in clean_test_concepts:
        if concept in pos_test_domain_concepts and concept in pos_train_relation_concepts:
            pos_test_relational_concepts.append(concept)
    print(f"Number of clean positive test relational concepts: {len(pos_test_relational_concepts)}")
    pos_test_relational_concepts_count = {}
    for concept in pos_test_relational_concepts:
        if concept in pos_test_domain_concepts:
            pos_test_relational_concepts_count[concept] = pos_train_domain_queries[concept]
        else:
            pos_test_relational_concepts_count[concept] = 0
    pos_test_relational_concepts = [concept for concept in pos_test_relational_concepts if pos_test_relational_concepts_count[concept] > 5]
    print(pos_test_relational_concepts_count)
    print(f"Number of clean positive test relational concepts that appear more than 5 times: {len(pos_test_relational_concepts)}")
    print(pos_test_relational_concepts)
    # remove "_Object" from pos_test_relational_concepts
    pos_test_relational_concepts = [concept.replace('_Object', '') for concept in pos_test_relational_concepts]
    print(len(pos_test_relational_concepts))
    print(pos_test_relational_concepts)
    # print positive test relational queries
    # for query in all_queries['test']['pos']:
    #     domain = create_domain_from_query([query])
    #     query_concepts = list(domain.functions.keys())[3:]
    #     if any(test_concept+'_Object_Object' in query_concepts for test_concept in pos_test_relational_concepts):
    #         print(query)
    print(f"Number of failed samples: {len(fail_samples)}")
    for i, sample in enumerate(tqdm(wave_dataset)):
        image = sample['img']
        config = sample['config.json']
        is_test = config['rep']
        subj = sample['__key__'].split('sub-')[1].split('_')[0]
        visual_voxel = sample['voxels']
        brain_region = sample['inputs']
        img_name = sample['img_name']
        # find scene_graph which has the same image name
        scene_graph = scene_graph_df[scene_graph_df['img_name'] == img_name]['scene_graph'].values[0]
        # remove characters like * and #
        scene_graph = scene_graph.replace('*', '')
        scene_graph = scene_graph.replace('#', '')
        scene_graph = scene_graph.replace('-', '')
        # remove empty lines
        scene_graph = '\n'.join([line for line in scene_graph.split('\n') if line.strip() != ''])
        scene_graph = scene_graph.lower()
        try:
            pos_scene_graph = scene_graph.split('negative samples')[0].strip()
            neg_scene_graph = scene_graph.split('negative samples')[1].strip()
        except:
            fail_samples.append(img_name)
            print(f'fail to parse {img_name}')
            print(scene_graph)
            continue
        _pos_queries = generate_queries(pos_scene_graph)
        # remove failed queries
        pos_queries = []
        for query in _pos_queries:
            domain = create_domain_from_query([query])
            parser = create_default_parser(domain)
            if len(domain.functions.keys()) > 3:
                concepts = list(domain.functions.keys())[3:]
                try:
                    x = parser.parse_expression(query)
                    if any(concept in concepts for concept in test_concepts_count):
                        continue
                    else:
                        pos_queries.append(query)
                except Exception as e:
                    print(f'fail to parse {query} {e}')
                    fail_samples.append(img_name)
        _neg_queries = generate_queries(neg_scene_graph)
        neg_queries = []
        for query in _neg_queries:
            domain = create_domain_from_query([query])
            parser = create_default_parser(domain)
            if len(domain.functions.keys()) > 3:
                concepts = list(domain.functions.keys())[3:]
                try:
                    x = parser.parse_expression(query)
                    if any(concept in concepts for concept in test_concepts_count):
                        continue
                    else:
                        neg_queries.append(query)
                except Exception as e:
                    print(f'fail to parse {query} {e}')
                    fail_samples.append(img_name)
        all_queries['train']['pos'].extend(pos_queries) if not is_test else all_queries['test']['pos'].extend(pos_queries)
        all_queries['train']['neg'].extend(neg_queries) if not is_test else all_queries['test']['neg'].extend(neg_queries)

        # generate description queries
        # TODO(Yanchen Wang @ 2025/03/30): add description queries
        description_queries = []
        description_answers = []
        for query in pos_queries:
            domain = create_domain_from_query([query])
            # if there exists relational attribute, add describe function
            attribute_concepts, relational_concepts, multi_relation_concepts = read_concepts_v2(domain=domain)
            if len(relational_concepts) > 0 and relational_concepts[0].replace('_Object', '') in pred_2_cls and relational_concepts[0].replace('_Object', '') in pos_test_relational_concepts:
                describe_query, describe_answer = generate_describe_queries(query, metaconcept=pred_2_cls[relational_concepts[0].replace('_Object', '')])
                if describe_query is not None:
                    description_queries.extend(describe_query)
                    print(f'{query} -> {describe_query}')
                    describe_domain = create_domain_from_query([describe_query])
                    description_categories = read_description_categories(domain=describe_domain)
                    description_answers.extend(describe_answer)
        # assert len(pos_queries) > 0, 'no positive queries'
        # assert len(neg_queries) > 0, 'no negative queries'
        # assert len(pos_queries + neg_queries + description_queries) > 0, 'no queries'
        queries = pos_queries + neg_queries + description_queries
        answers = ['yes' for _ in pos_queries] + ['no' for _ in neg_queries] + description_answers
        data_dict = {
            'image': image.numpy(),
            'subj': subj,
            'visual_voxel': visual_voxel.numpy(),
            'brain_region': brain_region,
            't_r': 2.0,
            'queries': queries,
            'answers': answers,
            **config
        }
        
        save_path = os.path.join(save_dir, subj)
        if is_test:
            save_path = os.path.join(save_path, 'test')
        else:
            save_path = os.path.join(save_path, 'train')
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f'{i}.npy'), data_dict)
    
if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    main(args)