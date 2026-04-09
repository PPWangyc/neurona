import os

import numpy as np
import torch


def get_npy_paths(data_dir, subject, split):
    # get all npy files in the data_dir
    split_dir = os.path.join(data_dir, subject, split)
    npy_paths = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.npy')]
    return npy_paths

# fmri question answering
class FQADataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir,
            npy_paths=None,
            subject='CSI1',
            split='train',
            transform=None,
            query_fn=None,
            mode='all',
            remove_query_set=None,
            remove_npy_path_set=None
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.mode = mode
        assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"
        # load data
        if npy_paths is None:
            self.npy_paths = get_npy_paths(data_dir, subject, split)
        else:
            self.npy_paths = npy_paths
        self.return_all_queries = True
        self.query_fn = query_fn
        self.remove_query_set = remove_query_set
        self.remove_npy_path_set = remove_npy_path_set
        if remove_npy_path_set is not None:
            self.npy_paths = [npy_path for npy_path in self.npy_paths if npy_path not in remove_npy_path_set]
    def __len__(self):
        return len(self.npy_paths)
    
    def __getitem__(self, idx):
        npy_path = self.npy_paths[idx]
        data = np.load(npy_path, allow_pickle=True).item()
        if self.transform is not None:
            data = self.transform(data)
        queries = data['queries']
        answers = data['answers']
        if self.mode == 'tf':
            queries = np.array(data['queries'])
            answers = np.array(data['answers'])
            # index of yes/no
            tf_idx = np.where((answers == 'yes') | (answers == 'no'))[0]
            # convert tf to list
            queries = queries[tf_idx].tolist()
            answers = answers[tf_idx].tolist()

        elif self.mode == 'all':
            pass
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.remove_query_set is not None:
            queries_ = []
            answers_ = []
            for query, answer in zip(queries, answers):
                # if self.split == 'train':
                if query not in self.remove_query_set:
                    queries_.append(query)
                    answers_.append(answer)
                        
                # else:
                #     if query in self.remove_query_set:
                #         queries_.append(query)
                #         answers_.append(answer)
                        
            queries = queries_
            answers = answers_
        num_queries = len(queries)
        if self.query_fn is not None:
            for i in range(num_queries):
                queries[i] = self.query_fn(queries[i])
        if len(queries) == 0:
            return {'path': npy_path}
        random_query_idx = np.random.randint(num_queries)
        if self.return_all_queries:
            return {
                **data,
                'queries': queries,
                'answers': answers,
            }
        else:
            return {
                **data,
                'queries': queries[random_query_idx],
                'answers': answers[random_query_idx],
            }
    