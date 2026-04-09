import json
import os
from copy import deepcopy
from typing import Optional

import concepts.benchmark.common.vocab
import jacinle
import numpy as np
import torch
from jacinle.config.environ_v2 import configs, set_configs
from tqdm import tqdm

from loader.fqa import FQADataset
from models.fmri.simple_cnn import SimplefMRIModel
from models.left.domain import create_domain_from_query, read_concepts_v2
from utils.dataset_utils import (convert_relation_to_attribute,
                                 create_object_object_variants,
                                 get_zs_train_dataset)
from utils.log_utils import logger
from utils.plot_utils import (evaluate_systematicity_results,
                              evaluate_vocab_results, evaluate_yes_no_results,
                              log_evaluation_result, plot_example_images)
from utils.utils import get_args, load_best_model, process_args, set_seed


def main(args):
    data_dir = args.data_dir
    if args.ground_rel_region and not args.ground_rel_relation:
        query_fn = convert_relation_to_attribute
        # query_fn = None
        # lr = 5e-4
    else:
        query_fn = None
    train_dataset = FQADataset(
        data_dir=data_dir,
        split='train',
        subject=args.subject,
        query_fn=query_fn,
        # mode='tf'
    )
    test_dataset = FQADataset(
        data_dir=data_dir,
        split='test',
        subject=args.subject,
        query_fn=query_fn,
        # mode='tf'
    )
    # load output vocab
    output_vocab = concepts.benchmark.common.vocab.Vocab().from_json('data/bold5000_output_vocab.json')
    # load vocab type
    vocab_type = json.load(open('data/bold5000_pred_cls.json'))
    # SET OUTPUT DIR
    args_dict = process_args(args)# SET OUTPUT DIR
    args_dict['vocab_type'] = vocab_type
    output_dir = args_dict['output_dir']
    logger.info(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    programs, train_programs, test_programs = [], [], []
    for i in tqdm(range(len(train_dataset))):
        train_programs.extend(train_dataset[i]['queries'])
    num_test_pos, num_test_neg, num_vocab = 0, 0, 0
    for i in tqdm(range(len(test_dataset))):
        test_programs.extend(test_dataset[i]['queries'])
        answers = test_dataset[i]['answers']
        # answers is a list of 'yes' or 'no'
        num_test_pos += sum(1 for ans in answers if ans == 'yes')
        num_test_neg += sum(1 for ans in answers if ans == 'no')
        num_vocab += sum(1 for ans in answers if ans in output_vocab.word2idx)
    
    if args.train_hop == 'zs':
        logger.info("Zero-shot training")
        train_dataset = get_zs_train_dataset(
            test_programs, 
            train_programs, 
            data_dir, 
            args.subject, 
            query_fn
        )
        train_programs= []
        for i in range(len(train_dataset)):
            train_programs.extend(train_dataset[i]['queries'])

    programs = train_programs + test_programs
    logger.info(f"Length of programs: {len(programs)}")
    logger.info(f"Test set: {num_test_pos} positive, {num_test_neg} negative, {num_vocab} vocab, {num_test_pos+num_test_neg+num_vocab} total")
    domain = create_domain_from_query(programs)

    # Create object_object variants for attribute concepts
    create_object_object_variants(domain) if args.ground_attr_relation and args.ground_rel_relation else None
    
    # Now you can check the new functions
    attribute_concepts, relation_concepts, multi_relation_concepts = read_concepts_v2(domain)
    logger.info(f"number of attribute concepts: {len(attribute_concepts)}, number of relation concepts: {len(relation_concepts)}, number of multi-relation concepts: {len(multi_relation_concepts)}")
    logger.info(f"Total domain functions: {len(domain.functions)}")
    train_domain=create_domain_from_query(train_programs)
    logger.info(f"Total train domain functions: {len(train_domain.functions)}")
    test_domain=create_domain_from_query(test_programs)
    logger.info(f"Total test domain functions: {len(test_domain.functions)}")
    # check how many test domain functions are not in train domain
    test_domain_concepts = set(test_domain.functions.keys())
    train_domain_concepts = set(train_domain.functions.keys())
    logger.warning(f"Test domain functions not in train domain: {test_domain_concepts - train_domain_concepts}") if len(test_domain_concepts - train_domain_concepts) > 0 else None
    logger.warning(f"Number of test domain functions not in train domain: {len(test_domain_concepts - train_domain_concepts)}") if len(test_domain_concepts - train_domain_concepts) > 0 else None
    train_dataset.return_all_queries = False
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    with set_configs():
        configs.model.domain = 'fqa'            # set to the name of the dataset
        configs.model.scene_graph = None            # set to the identifier of the scene graph extractor. In this case we will use our own implementation.
        configs.model.concept_embedding = 'linear-tied-attr'  # set to the identifier of the concept embedding. In this case we will use a simple linear mapping.
        configs.model.sg_dims = [None, 64, 64, 64]
        # set to the dimensions of the object and relation features. In this case we will use 64-dimensional features for all of them.
    
    model = SimplefMRIModel(
        domain=domain,
        parses=programs,
        # output_vocab=None,
        output_vocab=output_vocab,
        fmri_args=args_dict
    )
    model.to('cuda')
    # Define a helper function to update the meters.
    def update_meters(meters, monitors, prefix: Optional[str] = None):
        for k in list(monitors.keys()):
            if k + '/n' in monitors:
                meters.update({k: monitors[k]}, n=monitors[k + '/n'], prefix=prefix)
                del monitors[k]
                del monitors[k + '/n']

        meters.update(monitors, prefix=prefix)
    remove_untrain_concepts = True
    # Build the optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    group_meters = jacinle.GroupMeters()
    best_test_acc = 0
    train_hop = args.train_hop
    if train_hop in ['all', 'zs']:
        target_hop = [1,2,3]
    else:
        target_hop = [int(train_hop)]
    epochs = args.epochs
    # epochs = 0
    for epoch in range(epochs):
        model.train()
        group_meters.reset()
        for batch in jacinle.tqdm(train_loader, total=len(train_loader), desc='Train Epoch {}'.format(epoch)):
            queries = batch['queries']
            target_indices = []
            for i in range(len(queries)):
                query_domain = create_domain_from_query([queries[i]])
                query_concept = list(query_domain.functions.keys())[3:]
                if len(query_concept) in target_hop:
                    target_indices.append(i)
            if len(target_indices) == 0:
                continue
            batch['queries'] = [batch['queries'][i] for i in target_indices]
            batch['answers'] = [batch['answers'][i] for i in target_indices]
            batch['brain_region'] = batch['brain_region'][target_indices]
            # move batch to device
            batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss, monitors, outputs = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_meters(group_meters, {k: float(v) for k, v in monitors.items()})
            jacinle.get_current_tqdm().set_description(group_meters.format_simple(f'Epoch {epoch}', compressed=True))
        if epoch % 5 != 0 and epoch != epochs - 1:
            continue
        model.eval()
        group_meters.reset()
        with torch.no_grad():
            for data in jacinle.tqdm(test_dataset, total=len(test_dataset), desc='Test Epoch {}'.format(epoch)):
                data = {k: torch.Tensor(v).to('cuda').unsqueeze(0) if isinstance(v, np.ndarray) else v for k, v in data.items()}
                _data = deepcopy(data)
                for i in range(len(data['queries'])):
                    query_domain = create_domain_from_query([data['queries'][i]])
                    query_concept = list(query_domain.functions.keys())[3:]
                    if any([c not in train_domain_concepts for c in query_concept]) and remove_untrain_concepts:
                        logger.info(f"Skip query {data['queries'][i]} because it contains untrained concepts")
                        continue
                    _data['queries'] = [data['queries'][i]]
                    _data['answers'] = [data['answers'][i]]
                    outputs = model(_data)
                    update_meters(group_meters, {k: float(v) for k, v in outputs['monitors'].items()})
                jacinle.get_current_tqdm().set_description(group_meters.format_simple(f'Epoch {epoch} (test)', compressed=True))
            for k, v in group_meters.items():
                if k == 'acc/qa':
                    test_acc = v.avg
                    if test_acc > best_test_acc:
                        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_{}_{}.pth'.format(args.train_hop, args.atlas)))
                        logger.info(f"Save best model with test acc: {test_acc} at epoch {epoch}")
                        best_test_acc = test_acc
        logger.info(f"Epoch {epoch} test acc: {test_acc}, best test acc: {best_test_acc}")
    # load best model
    model = load_best_model(model, output_dir, args.train_hop, args.atlas)
    logger.info("Loaded best model for evaluation")

    # Evaluate yes/no queries and plot confusion matrix
    evaluate_yes_no_results(
        model, 
        test_dataset, 
        output_dir, 
        train_domain_concepts if args.train_hop != 'zs' else None,
        model_name='ns'
    )
    # Evaluate vocabulary-based queries and plot confusion matrix
    evaluate_vocab_results(
        model,
        test_dataset,
        output_dir,
        vocab_type,
        train_domain_concepts if args.train_hop != 'zs' else None,
        model_name='ns'
    )
    
    # Evaluate systematicity (argument swapping, predicate transfer, role systematicity)
    evaluate_systematicity_results(
        model,
        test_dataset,
        train_programs,
        output_dir,
        train_domain_concepts if args.train_hop != 'zs' else None,
        model_name='ns'
    )

    log_evaluation_result(output_dir)
    # Plot example images
    plot_example_images(model, test_dataset, os.path.join(output_dir, "example_images")) if args.plot_example_images else None
    logger.info("Done")

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    main(args)