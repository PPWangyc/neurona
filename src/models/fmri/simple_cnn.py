from itertools import product
from typing import Any, Dict, List, Optional, Union

import jacinle
import pandas as pd
import torch
import torch.nn as nn
from jacinle.config.environ_v2 import configs, set_configs

from models.left.models.model import LeftModel

ATLAS_PATH='data/difumo1024_yeo17_label.csv'
SCHAEFFER_1000_YEONETWORK_PATH = 'data/schaefer1000_yeo17_label.csv'
def update_from_loss_module(monitors, output_dict, loss_update):
    tmp_monitors, tmp_outputs = loss_update
    monitors.update(tmp_monitors)
    output_dict.update(tmp_outputs)

def load_atlas(atlas_path, atlas_name='Yeo_networks17'):
    if atlas_name == 'whole_brain':
        return {'whole_brain': list(range(1024))}
    atlas = pd.read_csv(atlas_path)
    # Create mapping from Yeo network to component indices
    yeo_groups = atlas.groupby(atlas_name)['Component'].apply(list).to_dict()
    # the values are lists, and make sure -1 to match the 0-based index
    for k, v in yeo_groups.items():
        yeo_groups[k] = [i - 1 for i in v]
    return yeo_groups

def convert_atlas_name(atlas_name):
    if atlas_name == 'yeo17':
        return 'Yeo_networks17'
    elif atlas_name == 'yeo7':
        return 'Yeo_networks7'
    elif atlas_name == 'difumo1024':
        return 'Difumo_names'
    elif atlas_name == 'whole':
        return 'whole_brain'
    else:
        raise ValueError(f'Unknown atlas name: {atlas_name}')

class fMRIRegionEmbedding(nn.Module):
    def __init__(self, atlas='yeo17', fmri_args: Optional[Dict[str, Any]] = None, removed_regions: Optional[List[str]] = None, selected_regions: Optional[List[str]] = None):
        super().__init__()
        self.fmri_args = fmri_args
        atlas_name = convert_atlas_name(atlas)
        if fmri_args['dataset'] == 'BOLD5000':
            _atlas_path = ATLAS_PATH
        else:
            _atlas_path = SCHAEFFER_1000_YEONETWORK_PATH
        self.yeo_groups = load_atlas(_atlas_path, atlas_name)
        
        # Handle selected_regions (only use specified regions) - takes precedence over removed_regions
        if selected_regions is not None:
            selected_regions_set = set(selected_regions)
            # Filter to only keep selected regions
            self.yeo_groups = {k: v for k, v in self.yeo_groups.items() if k in selected_regions_set}
            if len(selected_regions_set) > 0:
                print(f"Selected regions: {selected_regions_set}")
                print(f"Using regions: {list(self.yeo_groups.keys())}")
        # Remove specified regions if provided (only if selected_regions is not set)
        elif removed_regions is not None:
            removed_regions_set = set(removed_regions)
            # Filter out removed regions
            self.yeo_groups = {k: v for k, v in self.yeo_groups.items() if k not in removed_regions_set}
            if len(removed_regions_set) > 0:
                print(f"Removed regions: {removed_regions_set}")
                print(f"Remaining regions: {list(self.yeo_groups.keys())}")
        
        # Linear layer for each region
        network_dict = {}
        for k, v in self.yeo_groups.items():
            network_dict[k] = nn.Linear(len(v), 128)
        self.network_dict = nn.ModuleDict(network_dict)
        self.lenet = nn.Sequential(           # input is 1x32x32
            nn.Conv1d(5, 32, 4), nn.ReLU(),   # conv1: 32x28x28
            nn.MaxPool1d(2,2),               # pool1: 32x14x14
            nn.Conv1d(32, 32, 4), nn.ReLU(),  # conv2: 32x10x10
            nn.MaxPool1d(2, 2),               # pool2: 32x5x5
        )
        self.mlp = nn.Sequential(
            nn.Linear(640, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        # self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(32 * 29, 64), nn.ReLU())
        self.position_embedding = nn.Embedding(64, 64) # 64 positions, each with a 64-dimensional embedding
        self.fc2 = nn.Linear(256, 64)
        self.multi_relation_dense_layer = nn.Sequential(
            nn.ReLU(), nn.Linear(64, 64)
        )

    def forward(self, fmri):
        obj = self.forward_object(fmri)
        rel = self.forward_relation(obj)
        return obj, rel

    def forward_object(self, fmri):
        # each smail image is 32x32
        # img = B, 3, 32, 96, contains 3 small images in a row
        try:
            b = fmri.size(0)
        except:
            print(fmri.shape)
            print(type(fmri))
            exit()
        input_dict = {}
        for k, v in self.yeo_groups.items():
            input_dict[k] = fmri[..., v]
        # forwards through each network
        output_dict = {}
        for k, v in input_dict.items():
            network_features = self.network_dict[k](v)
            # output_dict[k] = network_features.flatten(1)
            # print(f'network {k} shape: {network_features.shape}')
            output_dict[k] = network_features
        # combine the outputs
        object_features = torch.stack([output_dict[k] for k in output_dict.keys()], dim=1)
        seq_len, channel, dim = object_features.size(1), object_features.size(2), object_features.size(3)
        object_features = object_features.reshape((b * seq_len, channel, dim))
        features = self.lenet(object_features)
        features = features.reshape((b, seq_len, -1))
        features = self.fc(features)
        return features

    def forward_relation(self, object_feat):
        nr_objects = object_feat.size(1)
        position_feature = torch.arange(nr_objects, dtype=torch.int64, device=object_feat.device)
        position_feature = self.position_embedding(position_feature)
        position_feature = position_feature.unsqueeze(0).expand(object_feat.size(0), nr_objects, 64)
        # combine object features and position features
        # position_feature = B, 3, 64 | object_feat = B, 3, 64
        feature = torch.cat([object_feat, position_feature], dim=-1)
        # feature = B, 3, 128
        # repeat the features for all objects
        feature1 = feature.unsqueeze(1).expand(feature.size(0), nr_objects, nr_objects, 128)
        # repeat the features for each object
        feature2 = feature.unsqueeze(2).expand(feature.size(0), nr_objects, nr_objects, 128)
        # form a feature matrix for each pair of objects
        feature = torch.cat([feature1, feature2], dim=-1)
        # feature = B, 3, 3, 256

        # apply a linear layer to get the relation features
        feature = self.fc2(feature)
        # feature = B, nr_objects, nr_objects, 64
        return feature
    
    def forward_triplet(self, object_feat):
        nr_objects = object_feat.size(1)
        position_feature = torch.arange(nr_objects, dtype=torch.int64, device=object_feat.device)
        position_feature = self.position_embedding(position_feature)
        position_feature = position_feature.unsqueeze(0).expand(object_feat.size(0), nr_objects, 64)
        # combine object features and position features
        feature = torch.cat([object_feat, position_feature], dim=-1)
        print(feature.shape)
        # feature = B, 3, 128

        rel = self.forward_relation(object_feat)
        print(rel.shape)
        relation_feat = [[0 for i in range(nr_objects)] for j in range(nr_objects)]
        multi_relation_feat = [[0 for i in range(nr_objects)] for j in range(nr_objects)]
        for i, j in product(range(nr_objects), range(nr_objects)):
            print(i, j)
        exit()
        rel = rel.unsqueeze(1).expand(rel.size(0), nr_objects, nr_objects, 64)
        print(rel.shape)
        exit()

        # apply a linear layer to get the relation features
        feature = self.fc2(feature)
        # feature = B, nr_objects, nr_objects, nr_objects, 64
        return feature   

with set_configs():
    configs.model.domain = 'fqa'            # set to the name of the dataset
    configs.model.scene_graph = None            # set to the identifier of the scene graph extractor. In this case we will use our own implementation.
    configs.model.concept_embedding = 'linear'  # set to the identifier of the concept embedding. In this case we will use a simple linear mapping.
    configs.model.sg_dims = [None, 64, 64, 64]  # set to the dimensions of the object and relation features. In this case we will use 64-dimensional features for all of them.

class SimplefMRIModel(LeftModel):
    def __init__(self, domain, parses: Dict[str, Union[str, List[str]]], output_vocab, fmri_args: Dict[str, Any]):
        super().__init__(domain, output_vocab, fmri_args)

        self.parses = parses
        removed_regions = fmri_args.get('removed_regions', None)
        selected_regions = fmri_args.get('selected_regions', None)
        self.scene_graph = fMRIRegionEmbedding(atlas=fmri_args['atlas'], fmri_args=fmri_args, removed_regions=removed_regions, selected_regions=selected_regions)

    def forward(self, feed_dict):
        feed_dict = jacinle.GView(feed_dict)
        monitors, outputs = {}, {}

        f_sng = self.forward_sng(feed_dict)
        for i in range(len(feed_dict.queries)):
            # initialize the grounding module
            grounding = self.grounding_cls(f_sng[i], self, self.training, apply_relation_mask=True, fmri_args=self.fmri_args)
            # question = feed_dict.question[i]
            # question i in the batch, string
            # raw_parsing = self.parses[question]
            raw_parsing = feed_dict.queries[i]
            # This will write to outputs['executions']
            try:
                # print(f"Executing {raw_parsing}")
                self.execute_program_from_queries(raw_parsing, grounding, outputs)
                # print(f"Successfully executed {raw_parsing}")
            except Exception as e:
                print(f"Failed to execute {raw_parsing}")
                print(e)
                exit()
        update_from_loss_module(monitors, outputs, self.qa_loss(outputs['executions'], feed_dict.answers, question_types=['bool' for _ in outputs['executions']]))
        # print(monitors)
        # print(f'------------------------------------')
        # print(outputs['parsings'][0])
        # print(outputs['execution_traces'][0])
        # print(len(outputs['execution_traces'][0]))
        # exit()
        # print(type(outputs['execution_traces'][0][1]))
        # print(len(outputs['execution_traces'][0][1]))
        # print(outputs['execution_traces'][0][1][1].tensor.cpu().numpy())
        # print(f'------------------------------------')
        if self.training:
            loss = monitors['loss/qa']
            return loss, monitors, outputs
        else:
            outputs['monitors'] = monitors
            return outputs

    def forward_sng(self, feed_dict):
        object_features, relation_features = self.scene_graph(feed_dict.brain_region)
        all_scene_features = [{'attribute': object_features[i], 'relation': relation_features[i]} for i in range(len(object_features))]
        return all_scene_features
    
def get_concept_grounding_tensor(outputs):
    assert 'execution_traces' in outputs and 'parsings' in outputs, 'Outputs should contain execution traces and parsings'
    assert len(outputs['execution_traces']) == 1, 'Only support batch size 1'
    return outputs['execution_traces'][0][1][1].tensor

