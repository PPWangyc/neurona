import os

import numpy as np
from tqdm import tqdm

from loader.fqa import FQADataset
from models.left.domain import (BOOL, Function, FunctionTyping, ObjectType,
                                create_domain_from_query)


def replace_concepts(concept):
    concept = concept.replace(' ', '_')
    concept = concept.replace('-', '_')
    concept = concept.replace('.', '')
    concept = concept.replace('\'', '')
    if concept == 'in':
        concept = '_in'
    if concept == 'with':
        concept = '_with'
    if concept == 'and':
        concept = '_and'
    if concept == 'or':
        concept = '_or'
    if concept == 'from':
        concept = '_from'
    if concept == 'is':
        concept = '_is'
    # fix sit and sits
    if concept == 'sits':
        concept = 'sit'
    if concept == 'sitting':
        concept = 'sit'
    # fix have, contains, contain
    if concept == 'have':
        concept = 'has'
    if concept == 'contains':
        concept = 'has'
    if concept == 'contain':
        concept = 'has'
    # fix above,
    if concept == 'above':
        concept = 'on'
    if concept == 'near':
        concept = 'next_to'
    if concept == 'at':
        concept = '_in'
    if concept == 'carry':
        concept = 'hold'
    if concept =='wait':
        concept = 'resting'
    if concept == 'perch':
        concept = 'resting'
    if concept == 'around':
        concept = 'surround'
    if concept == 'standing':
        concept = 'stand'
    if concept =='perform':
        concept = 'display'
    if concept == 'drink':
        concept = 'eat'
    if concept == 'behind':
        concept = 'beside'

    return concept

def generate_object_exists(subgraph_str):
    obj = subgraph_str.split('<object>')[-1].split('</object>')[0]
    obj = replace_concepts(obj)
    obj_exists = 'exists(Object, lambda x: ' + obj + '(x))'
    return obj_exists

def generate_descriptor_object_exists(subgraph_str):
    adj = subgraph_str.split('<adjective>')[-1].split('</adjective>')[0]
    adj = replace_concepts(adj)
    sub = subgraph_str.split('<subject>')[-1].split('</subject>')[0]
    sub = replace_concepts(sub)
    adj_sub_exists = 'exists(Object, lambda x: ' + adj + '(x) and ' + sub + '(x))'
    return adj_sub_exists

def generate_relation_object_exists(subgraph_str):
    sub = subgraph_str.split('<subject>')[-1].split('</subject>')[0]
    sub = replace_concepts(sub)
    pred = subgraph_str.split('<predicate>')[-1].split('</predicate>')[0]
    pred = replace_concepts(pred)
    obj = subgraph_str.split('<object>')[-1].split('</object>')[0]
    obj = replace_concepts(obj)
    if pred == 'belong_to':
        _sub = sub
        sub = obj
        obj = _sub
        pred = 'has'
    rel_object_exists = 'exists(Object, lambda x: ' + sub + '(x) and ' + pred + '(x, iota(Object, lambda y: ' + obj + '(y))))'
    return rel_object_exists

# Function to generate positive queries
def generate_queries(scene_graph):
    positive_queries = []
    # make scene graph a list of subgraphs by splitting on \n
    subgraphs = scene_graph.split('\n')
    for subgraph in subgraphs:
        if '<relation>' in subgraph:
            positive_queries.append(generate_relation_object_exists(subgraph))
        elif '<descriptor>' in subgraph and '<subject>' in subgraph:
            positive_queries.append(generate_descriptor_object_exists(subgraph))
        elif '<object>' in subgraph:
            positive_queries.append(generate_object_exists(subgraph))
        else:
            print('No query generated for subgraph: ', subgraph)
    return positive_queries

POSITION_REVERSE_MAPPING = {
    'on': 'under',
    'in_front_of': 'beside',
    'under': 'on',
    'beside': 'in_front_of',
}

def generate_describe_queries(query, metaconcept='relation'):
    # check if the query is a relation
    assert 'iota(' in query
    sub = query.split('exists(Object, lambda x: ')[-1].split('(x)')[0]
    pred = query.split('(x) and ')[-1].split('(x, iota')[0]
    obj = query.split('iota(Object, lambda y: ')[-1].split('(y))')[0]
    describe_query_list = []
    answer_list = []
    if metaconcept.capitalize() == 'Movement':
        describe_query = f'describe({metaconcept.capitalize()}, lambda k: {metaconcept.lower()}(k, iota(Object, lambda x: {sub}(x)) and iota(Object, lambda y: {obj}(y))))'
        answer = pred
        describe_query_list.append(describe_query)
        answer_list.append(answer)
    elif metaconcept.capitalize() == 'Position':
        describe_query = f'describe({metaconcept.capitalize()}, lambda k: {metaconcept.lower()}(k, iota(Object, lambda x: {sub}(x)) and iota(Object, lambda y: {obj}(y))))'
        answer = pred
        describe_query_list.append(describe_query)
        answer_list.append(answer)
        if pred in POSITION_REVERSE_MAPPING:
            describe_query = f'describe({metaconcept.capitalize()}, lambda k: {metaconcept.lower()}(k, iota(Object, lambda x: {obj}(x)) and iota(Object, lambda y: {sub}(y))))'
            answer = POSITION_REVERSE_MAPPING[pred]
            describe_query_list.append(describe_query)
            answer_list.append(answer)
    else:
        return None, None
    return describe_query_list, answer_list

def convert_relation_to_attribute(relation_str):
    if not 'x, iota(Object, lambda' in relation_str:
        return relation_str
    pred = relation_str.split('(x) and ')[-1].split('(x, iota')[0]
    sub = relation_str.split('exists(Object, lambda x: ')[-1].split('(x)')[0]
    obj = relation_str.split('iota(Object, lambda y: ')[-1].split('(y))')[0]
    # if any pred, sub or obj is empty, return the original relation
    if pred == '' or sub == '' or obj == '':
        return relation_str
    attribute_str = 'exists(Object, lambda x: exists(Object, lambda y: exists(Object, lambda z: ' + sub + '(x) and ' + pred + '(y) and ' + obj + '(z))))'
    return attribute_str

def symbolic_to_natural_language(query):
    if 'exists' in query:
        return exists_to_natural_language(query)
    elif 'describe' in query:
        return describe_to_natural_language(query)
    else:
        return "Unsupported query type"

def exists_to_natural_language(query):
    if 'iota' in query:
        return exists_to_natural_language_with_relation(query)
    else:
        return exists_to_natural_language_with_attribute(query)

def describe_to_natural_language(query):
    subject = query.split('iota(Object, lambda x: ')[-1].split('(x)')[0]
    pred = query.split('lambda k: ')[-1].split('(k, iota')[0]
    obj = query.split('iota(Object, lambda y: ')[-1].split('(y))')[0]
    return f"What is the {pred} of the {subject} and {obj}?"

def exists_to_natural_language_with_relation(query):
    subject = query.split('exists(Object, lambda x: ')[-1].split('(x)')[0]
    pred = query.split('(x) and ')[-1].split('(x, iota')[0]
    obj = query.split('iota(Object, lambda y: ')[-1].split('(y))')[0]
    return f"Is there a {subject} {pred} the {obj}?"

def exists_to_natural_language_with_attribute(query):
    if ' and ' in query:
        subject = query.split('exists(Object, lambda x: ')[-1].split('(x)')[0]
        obj = query.split(' and ')[-1].split('(x)')[0]
        return f"Is there a {subject} {obj}?"
    else:
        return f"Is there a {query.split('exists(Object, lambda x: ')[-1].split('(x)')[0]}?"

def get_tar_file_list(
    tar_dir: str,
    subject_list: list,
    split_list: list,
    ) -> list:
    """
    Get a list of tar files in the tar_dir.
    Args:
        tar_dir (str): The directory containing tar files.
        subject_list (list): List of subjects.
        split_list (list): List of splits.
    Returns:
        tar_file_list (list): List of tar files.
    """
    tar_file_list = []
    for subject in subject_list:
        for split in split_list:
            pwd = os.path.join(tar_dir, subject, split)
            tar_file_list += [os.path.join(pwd, f) for f in os.listdir(pwd) if f.endswith('.tar')]
    return tar_file_list

def stats_concepts(programs):
    stats = {}
    for program in tqdm(programs):
        query_domain = create_domain_from_query([program])
        query_concepts = list(query_domain.functions.keys())[3:]
        for concept in query_concepts:
            if concept in stats:
                stats[concept] += 1
            else:
                stats[concept] = 1
    return stats

def create_object_object_variants(domain):
    """
    Create Object_Object variants of existing functions in the domain.
    For each function that takes a single Object argument and returns a BOOL,
    create a new function that takes two Object arguments and returns a BOOL.
    """
    functions_to_add = []
    for func_name, func in domain.functions.items():
        if func_name.endswith('_Object_Object'):
            new_func_name = func_name[:-7]
            new_func = Function(
                new_func_name,
                FunctionTyping[BOOL](ObjectType('Object'))
            )
            functions_to_add.append(new_func)
        elif func_name.endswith('_Object'):
            new_func_name = func_name[:-7] + '_Object_Object'
            new_func = Function(
                new_func_name,
                FunctionTyping[BOOL](ObjectType('Object'), ObjectType('Object'))
            )
            functions_to_add.append(new_func)
        else:
            print(f"Skipping function: {func_name}")

    for func in functions_to_add:
        if func.name in domain.functions:
            print(f"Function {func.name} already exists")
            continue
        domain.define_function(func)

def parse_relation_query(query):
    """
    Parse a relation query to extract subject, predicate, and object.
    Returns:
        tuple: (subject, predicate, object) or None if not a relation query
    """
    if 'iota(' not in query:
        return None

    try:
        subject = query.split('exists(Object, lambda x: ')[-1].split('(x)')[0]
        pred = query.split('(x) and ')[-1].split('(x, iota')[0]
        obj = query.split('iota(Object, lambda y: ')[-1].split('(y))')[0]
        return (subject, pred, obj)
    except (IndexError, ValueError):
        return None

def classify_systematicity_queries(test_queries, train_queries):
    """
    Classify test queries into three types of systematicity:
    1. Argument swapping: same predicate, swapped objects
    2. Predicate transfer: same arguments, different predicate
    3. Role systematicity: combination of both (cross-product)
    """
    train_relations = {}
    train_by_subject_pred = {}
    train_by_predicate_obj = {}
    train_by_subject_obj = {}

    for query in train_queries:
        parsed = parse_relation_query(query)
        if parsed is not None:
            sub, pred, obj = parsed
            train_relations[(sub, pred, obj)] = query

            if (sub, pred) not in train_by_subject_pred:
                train_by_subject_pred[(sub, pred)] = set()
            train_by_subject_pred[(sub, pred)].add(obj)

            if (pred, obj) not in train_by_predicate_obj:
                train_by_predicate_obj[(pred, obj)] = set()
            train_by_predicate_obj[(pred, obj)].add(sub)

            if (sub, obj) not in train_by_subject_obj:
                train_by_subject_obj[(sub, obj)] = set()
            train_by_subject_obj[(sub, obj)].add(pred)

    classifications = {
        'argument_swapping': [],
        'predicate_transfer': [],
        'role_systematicity': [],
        'none': []
    }

    for test_query in test_queries:
        parsed = parse_relation_query(test_query)
        if parsed is None:
            classifications['none'].append(test_query)
            continue

        test_sub, test_pred, test_obj = parsed

        is_arg_swap = False
        if (test_sub, test_pred) in train_by_subject_pred:
            train_objects = train_by_subject_pred[(test_sub, test_pred)]
            if test_obj not in train_objects and len(train_objects) > 0:
                is_arg_swap = True

        is_pred_transfer = False
        if (test_sub, test_obj) in train_by_subject_obj:
            train_predicates = train_by_subject_obj[(test_sub, test_obj)]
            if test_pred not in train_predicates and len(train_predicates) > 0:
                is_pred_transfer = True

        is_role_systematicity = False
        if is_arg_swap and is_pred_transfer:
            train_objs_for_sp = train_by_subject_pred.get((test_sub, test_pred), set())
            train_preds_for_so = train_by_subject_obj.get((test_sub, test_obj), set())

            for train_obj in train_objs_for_sp:
                if train_obj != test_obj:
                    for train_pred in train_preds_for_so:
                        if train_pred != test_pred:
                            if (test_sub, train_pred, train_obj) in train_relations:
                                is_role_systematicity = True
                                break
                    if is_role_systematicity:
                        break

        if is_role_systematicity:
            classifications['role_systematicity'].append(test_query)
        elif is_arg_swap:
            classifications['argument_swapping'].append(test_query)
        elif is_pred_transfer:
            classifications['predicate_transfer'].append(test_query)
        else:
            classifications['none'].append(test_query)

    return classifications

def compute_systematicity_accuracy(test_queries, test_answers, test_pred_answers, train_queries):
    """
    Compute accuracy for each type of systematicity query.
    """
    def to_bool(ans):
        if isinstance(ans, str):
            return ans.lower() == 'yes'
        return bool(ans)

    test_answers_bool = [to_bool(a) for a in test_answers]
    test_pred_answers_bool = [to_bool(a) for a in test_pred_answers]

    classifications = classify_systematicity_queries(test_queries, train_queries)

    results = {}

    for category, queries in classifications.items():
        if len(queries) == 0:
            results[category] = {
                'accuracy': None,
                'total': 0,
                'correct': 0,
                'queries': []
            }
            continue

        query_set = set(queries)
        category_indices = [i for i, q in enumerate(test_queries) if q in query_set]

        if len(category_indices) == 0:
            results[category] = {
                'accuracy': None,
                'total': 0,
                'correct': 0,
                'queries': []
            }
            continue

        category_answers = [test_answers_bool[i] for i in category_indices]
        category_preds = [test_pred_answers_bool[i] for i in category_indices]
        correct = sum(1 for a, p in zip(category_answers, category_preds) if a == p)
        total = len(category_indices)
        accuracy = correct / total if total > 0 else None

        results[category] = {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'queries': queries
        }

    return results

def report_systematicity_accuracy(results, output_file=None):
    """
    Print and optionally save a report of systematicity accuracy results.
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("SYSTEMATICITY ACCURACY REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")

    categories = ['argument_swapping', 'predicate_transfer', 'role_systematicity', 'none']
    category_names = {
        'argument_swapping': 'Argument Swapping',
        'predicate_transfer': 'Predicate Transfer',
        'role_systematicity': 'Role Systematicity',
        'none': 'Other Queries'
    }

    for category in categories:
        if category in results:
            data = results[category]
            name = category_names.get(category, category)
            report_lines.append(f"{name}:")
            if data['accuracy'] is not None:
                report_lines.append(f"  Accuracy: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")
            else:
                report_lines.append(f"  Accuracy: N/A (no queries in this category)")
            report_lines.append(f"  Total queries: {data['total']}")
            report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to {output_file}")

def get_zs_train_dataset(test_programs, train_programs, data_dir, subject, query_fn=None):
    """
    Create a train dataset by removing queries that appear in both test and train sets.
    """
    test_queries_set = set(test_programs)
    train_queries_set = set(train_programs)

    intersection = test_queries_set & train_queries_set

    intersection_domain = create_domain_from_query(list(intersection))
    unique_train_queries = train_queries_set - intersection
    unique_train_domain = create_domain_from_query(list(unique_train_queries))

    intersection_concepts = set(intersection_domain.functions.keys())
    train_unique_concepts = set(unique_train_domain.functions.keys())
    intersection_only_concepts = intersection_concepts - train_unique_concepts

    train_dataset = FQADataset(
        data_dir=data_dir,
        split='train',
        subject=subject,
        query_fn=query_fn,
        remove_query_set=intersection,
    )

    remove_npy_path_set = set()
    for i in tqdm(range(len(train_dataset))):
        if 'path' in train_dataset[i]:
            remove_npy_path_set.add(train_dataset[i]['path'])

    train_dataset = FQADataset(
        data_dir=data_dir,
        split='train',
        subject=subject,
        query_fn=query_fn,
        remove_query_set=intersection,
        remove_npy_path_set=remove_npy_path_set,
    )

    return train_dataset
