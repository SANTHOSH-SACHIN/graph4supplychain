import streamlit as st
import urllib3
import glob
import json
import networkx as nx
import pandas as pd
from torch_geometric.data import HeteroData
import torch
from collections import defaultdict
import requests
from torch_geometric.utils.convert import from_networkx
import re
import os
from torch_geometric.data import Data
import torch_geometric.transforms as T
import numpy as np
import torch.nn.functional as F
import plotly.graph_objects as go
import torch
from torch_geometric.nn import to_hetero
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.nn.conv import GATConv , SAGEConv
from datetime import timedelta
import copy
import random

class TemporalHeterogeneousGraphParser:
    def __init__(self, base_url, version, headers,meta_data_path, use_local_files=False, local_dir=None,  num_classes =20, num_quartiles=4):
        self.base_url = base_url
        self.version = version
        self.headers = headers
        self.use_local_files = use_local_files
        self.local_dir = local_dir
        self.timestamps = []
        self.num_classes = num_classes
        self.meta_data_path = meta_data_path
        
        self.node_features = {}
        self.edge_features = {}
        self.edge_indices = {}
        self.node_label = {}
        self.class_ranges = {}

        self.df = None
        self.po_df = None
        self.df_date = None
        self.df_dict={}
        
        self.parts_dict = {}
        self.num_quartiles = num_quartiles
        
        if use_local_files and not local_dir:
            raise ValueError("Local directory must be specified when use_local_files is True.")

    def fetch_timestamps(self):
        if self.use_local_files:
            if os.path.exists(self.meta_data_path):
                with open(self.meta_data_path, 'r') as f:
                    self.metadata = json.load(f)
                self.timestamps = list(range(1, self.metadata.get('total_timestamps', 0)))
            else:
                raise FileNotFoundError(f"Metadata file not found at {self.meta_data_path}")
        else:
            schema_url = f"{self.base_url}/archive/schema/{self.version}"
            response = requests.get(schema_url, headers=self.headers, verify=False)
            
            if response.status_code == 200:
                self.timestamps = sorted(response.json())
            else:
                raise Exception(f"Failed to retrieve timestamps: HTTP {response.status_code}")
            if os.path.exists(self.meta_data_path):
                with open(self.meta_data_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}

            version_dir = os.path.join(self.local_dir, self.version)
            if os.path.exists(version_dir):
                self.use_local_files = True
            else:
                self.download_version_files(version_dir)
                self.use_local_files = True

    def fetch_json(self, timestamp):
        if self.use_local_files:
            file_path = os.path.join(self.local_dir, self.version, f"{timestamp}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                raise FileNotFoundError(f"File for timestamp {timestamp} not found: {file_path}")
        else:
            url = f"{self.base_url}/archive/schema/{self.version}/{timestamp}"
            response = requests.get(url, headers=self.headers, verify=False)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to retrieve data for timestamp {timestamp}: HTTP {response.status_code}")


    def download_version_files(self, version_dir):
        os.makedirs(version_dir, exist_ok=True)
        if not self.timestamps:
            raise ValueError("Timestamps are not initialized. Ensure `fetch_timestamps` is called before downloading files.")

        print(f"Downloading JSON files for version {self.version}...")
        for timestamp in self.timestamps:
            try:
                data = self.fetch_json(timestamp)
                file_path = os.path.join(version_dir, f"{timestamp}.json")
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                print(f"Successfully downloaded and saved: {file_path}")
            except Exception as e:
                print(f"Failed to download or save data for timestamp {timestamp}: {e}")
        print(f"All files for version {self.version} have been downloaded.")
    
    def get_edge_index(self, edge_type, str):
        return self.edge_schema[edge_type].index(str)
            
    def get_node_index(self, node_type, str):
        return self.node_schema[node_type].index(str)

    def get_node_attr(self, type):
        return self.metadata["node_types"][type.upper()]

    def get_edge_attr(self, type):
        return self.metadata["relationship_types"][type]

    def get_df(self):
        return self.df
    def get_po_df(self):
        return self.po_df

    def get_date_df(self):
        return self.df_date

    def set_df(self, timestamp):
        base = pd.Timestamp(self.metadata['base_date'])  
        date = (base + timedelta(days=timestamp)).strftime('%Y-%m-%d') 
        
        new_row = pd.DataFrame([self.demand], index=[date])
        if self.df is None:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row])

    def set_po_df(self, timestamp):
        base = pd.Timestamp(self.metadata['base_date'])  
        date = (base + timedelta(days=timestamp)).strftime('%Y-%m-%d') 
        val1_dict = {key: val[0] for key, val in self.po_demand.items()} 
        new_row = pd.DataFrame([val1_dict], index=[date])
        if self.po_df is None:
            self.po_df = new_row
        else:
            self.po_df = pd.concat([self.po_df, new_row]) 

    def set_date_df(self, timestamp, data):
        base = pd.Timestamp(self.metadata['base_date'])
        date = (base + timedelta(days=timestamp)).strftime('%Y-%m-%d')

        row_data = {}
    
        for row in data['node_values']['PARTS']:
            col_name = row[self.get_node_index('PARTS', 'id')]
            st = row[self.get_node_index('PARTS', 'valid_from')]
            exp = row[self.get_node_index('PARTS', 'valid_till')]
            row_data[col_name] = (st, exp)
    
        temp_df = pd.DataFrame([row_data], index=[date])
    
        if self.df_date is None:
            self.df_date = temp_df
        else:
            self.df_date = pd.concat([self.df_date, temp_df])

    def encode_categorical_value(self, key, value):
        allowed_values = self.metadata.get('allowed_values', {}).get(key, {})
        if isinstance(allowed_values, dict):
            for main_category, sub_values in allowed_values.items():
                if value == main_category:
                    return list(allowed_values.keys()).index(main_category) * 100  # Encode main category uniquely
                if value in sub_values:
                    return (list(allowed_values.keys()).index(main_category) * 100 +sub_values.index(value) + 1) 
        elif isinstance(allowed_values, list) and value in allowed_values:
            return allowed_values.index(value)
        return -1

    def aggregate_part_features(self, part_id, agg_method="mean"):
        complete_df = self.dfs
        agg_functions = {
            "mean": np.mean,
            "sum": np.sum,
            "min": np.min,
            "max": np.max
        }
        
        if agg_method not in agg_functions:
            raise ValueError(f"Invalid aggregation method: {agg_method}. Choose from {list(agg_functions.keys())}.")
    
        part_df = complete_df[part_id].copy()
    
        w_columns = [col for col in part_df.columns if col.startswith("W_")]
        f_columns = [col for col in part_df.columns if col.startswith("F_")]
    
        part_df["W_agg"] = part_df[w_columns].apply(
            lambda row: list(agg_functions[agg_method](np.array(row.tolist()), axis=0)), axis=1
        )
    
        part_df["F_agg"] = part_df[f_columns].apply(
            lambda row: list(agg_functions[agg_method](np.array(row.tolist()), axis=0)), axis=1
        )
        part_df = part_df.drop(columns=w_columns + f_columns)
        return part_df

    def store_feat(self,data):
        feat={}
        for n_type, t in data['node_values'].items():
            for n in t:
                lis=[]
                for i in (self.metadata['node_types'][n_type]):
                    idx = self.get_node_index(n_type, i)
                    if isinstance(n[idx], (int, float)):
                        lis.append(n[idx])
                feat[n[self.get_node_index(n_type, 'id')]] = lis
        return feat    

    def get_extended_df(self):
        return self.dfs
                
    def set_extended_df(self, data_dict, type=True):
        base_date = pd.Timestamp(self.metadata['base_date'])
        edge_list={}
        
        df = self.df
        
        allowed_edges=[]
        parts =[]

        key_dict={}
        feat={}

        for i in data_dict[1]['node_values'][type]:
            parts.append(i[self.get_node_index(type, 'id')])
        
        for edge_type in data_dict[1]['relationship_types']:
            edge_split = re.split(r'To', edge_type)
            if type in edge_split:
                allowed_edges.append(edge_type)


        for edge in data_dict[1]['link_values']:
            e_type=''
            for edge_type in data_dict[1]['relationship_types']:
                if edge_type in edge:
                    e_type=edge_type
            if e_type in allowed_edges:
                src = edge[self.get_edge_index(e_type, 'source')]
                dst = edge[self.get_edge_index(e_type, 'target')]
                if src in parts:
                    if src in key_dict:
                        key_dict[src].append(dst)
                    else:
                        key_dict[src] = [dst]
                else:
                    if dst in key_dict:
                        key_dict[dst].append(src)
                    else:
                        key_dict[dst] = [src]

        for ts, data in data_dict.items():
            feat = self.store_feat(data)
            lis={}
            lis[ts]=[]
            for i,j in key_dict.items():
                lis[ts].append(feat[i])
                for n in j:
                    lis[ts].append(feat[n])
                    
        # Dictionary to store DataFrames for each `i`
        dfs = {}
        cols={}
        for ts, data in data_dict.items():
            feat = self.store_feat(data)
            lis = {}
            lis[ts] = []
            date = (base_date + timedelta(days=ts)).strftime('%Y-%m-%d') 
            
            for i, j in key_dict.items():
                if i not in dfs:
                    dfs[i] = pd.DataFrame()
                cols[i]=[i]
                row_data = [feat[i]]
                row_data.extend([feat[n] for n in j])
                cols[i].extend(list(j))
                lis[ts].extend(row_data)
                dfs[i] = pd.concat(
                    [dfs[i], pd.DataFrame([row_data], index=[date])], axis=0
                )
        for i, df in dfs.items():
            if len(cols[i]) == df.shape[1]:
                df.columns = cols[i]
            else:
                raise ValueError(f"Mismatch: cols[{i}] has {len(cols[i])} columns, but DataFrame has {df.shape[1]} columns.")
        self.dfs=dfs

    def multistep_data(self, temporal_graphs, out_steps):
        timestamps = sorted(temporal_graphs.keys())
        
        for ts, graph in temporal_graphs.items():
            graph = graph[1]
        
            for node_type in graph.node_types:
                if 'y' in graph[node_type]: 
                    num_nodes = graph[node_type]['y'].shape[0]
                    original_dtype = graph[node_type]['y'].dtype
                    temporal_y = torch.zeros((num_nodes, out_steps), 
                                             device=graph[node_type]['y'].device, 
                                             dtype=original_dtype)  # Retain the dtype
                    latest_y = graph[node_type]['y']
                    for step in range(out_steps):
                        next_ts = ts + step  
                        if next_ts in temporal_graphs:
                        
                            next_graph = temporal_graphs[next_ts][1]
                            if 'y' in next_graph[node_type]:
                                latest_y = next_graph[node_type]['y']
                        
                        temporal_y[:, step] = latest_y.squeeze(-1)
                    graph[node_type]['y'] = temporal_y
        
        return temporal_graphs

    def compute_demand(self, data, threshold):
            po_demand={}
            facility_demand={}
            facility_external_demand={}
            sa_demand={}
            raw_demand={}
            edges={}
            edges_2={}
            for po in data['node_values']['PRODUCT_OFFERING']:
                po_demand[po[self.get_node_index('PRODUCT_OFFERING', 'id')]] = [po[self.get_node_index('PRODUCT_OFFERING', 'demand')], 0, 0]
    
            for edge in data['link_values']:
                ind = self.get_edge_index("FACILITYToPRODUCT_OFFERING", 'relationship_type')
                if ind>=len(edge):
                    continue
                if edge[ind] == "FACILITYToPRODUCT_OFFERING":
                    target_ind = self.get_edge_index("FACILITYToPRODUCT_OFFERING", 'target')
                    source_ind = self.get_edge_index("FACILITYToPRODUCT_OFFERING", 'source')
                    po_demand[edge[target_ind]][2]+=1
    
                    source = edge[source_ind]
    
                    if source in edges:
                        edges[edge[source_ind]].append(edge)
                    else:
                        edges[edge[source_ind]] = edge
                        
                if edge[self.get_edge_index("FACILITYToPARTS", 'relationship_type')] == "FACILITYToPARTS":
                    target_ind = self.get_edge_index("FACILITYToPARTS", 'target')
                    source_ind = self.get_edge_index("FACILITYToPARTS", 'source')
                    source = edge[source_ind]
    
                    if source in edges:
                        edges_2[edge[source_ind]].append(edge)
                    else:
                        edges_2[edge[source_ind]] = edge
                        
            bottle_neck={}
            supply={}
            for facility in data['node_values']['FACILITY']:
                if facility[self.get_node_index('FACILITY','type')] == "lam":
                    id = facility[self.get_node_index('FACILITY','id')]
                    target = edges[id][self.get_edge_index('FACILITYToPRODUCT_OFFERING', 'target')]
                    po_demand[target][1]+= facility[self.get_node_index('FACILITY','max_capacity')]
    
            for facility in data['node_values']['FACILITY']:
                if facility[self.get_node_index('FACILITY','type')] == "lam":
                    id = facility[self.get_node_index('FACILITY','id')]
                    target = edges[id][self.get_edge_index('FACILITYToPRODUCT_OFFERING', 'target')]
                    dem = (po_demand[target][0]/po_demand[target][1])*facility[self.get_node_index('FACILITY','max_capacity')]
                    supply[id] = po_demand[target][1]
                    facility_demand[id] = dem
                    if po_demand[target][0]/po_demand[target][2] > dem + threshold:
                        bottle_neck[id] = 1
                    else:
                        bottle_neck[id] = 0
    
            for edge in data['link_values']:
                ind = self.get_edge_index("PARTSToFACILITY", 'relationship_type')
                if ind>=len(edge):
                    continue
                if edge[ind] == "PARTSToFACILITY":
                    if edge[self.get_edge_index("PARTSToFACILITY", 'target')] in facility_demand:
                        src = edge[self.get_edge_index("PARTSToFACILITY", 'source')]
                        if src in sa_demand:
                            sa_demand[src][0] += edge[self.get_edge_index("PARTSToFACILITY", 'quantity')] * facility_demand[edge[self.get_edge_index("PARTSToFACILITY", 'target')]]
                        else:
                            sa_demand[src] = [edge[self.get_edge_index("PARTSToFACILITY", 'quantity')] * facility_demand[edge[self.get_edge_index("PARTSToFACILITY", 'target')]], 0, 0]
                        
    
            list_pop=[]
            for parts in data['node_values']['PARTS']:
                source=parts[self.get_node_index('PARTS','id')]
                if parts[self.get_node_index('PARTS','type')]=='subassembly':
                    if source not in sa_demand:
                        list_pop.append(source)
    
            
            for facility in data['node_values']['FACILITY']:
                if facility[self.get_node_index('FACILITY','type')] == 'external':
                    id = facility[self.get_node_index('FACILITY','id')]
                    target = edges_2[id][self.get_edge_index('FACILITYToPARTS', 'target')]
                    if target in sa_demand:
                        sa_demand[target][1] += facility[self.get_node_index('FACILITY','max_capacity')]
                        sa_demand[target][2] += 1
                    else:
                        if target not in list_pop:
                            list_pop.append(target)
                        if id not in list_pop:
                            list_pop.append(id)
                        
    
            
            for facility in data['node_values']['FACILITY']:
                if facility[self.get_node_index('FACILITY','type')] == "external":
                    id = facility[self.get_node_index('FACILITY','id')]
                    target = edges_2[id][self.get_edge_index('FACILITYToPARTS', 'target')]
                    if target in list_pop:
                        if id not in list_pop:
                            list_pop.append(id)
                    else:
                        dem =  (sa_demand[target][0]/  sa_demand[target][1])* facility[self.get_node_index('FACILITY','max_capacity')]
                        facility_external_demand[id] = dem
                        supply[id] = sa_demand[target][1]
                        if sa_demand[target][0]/  sa_demand[target][2] > dem + threshold:
                            bottle_neck[id] = 1
                        else:
                            bottle_neck[id] = 0
            
            for edge in data['link_values']:
                ind = self.get_edge_index("PARTSToFACILITY", 'relationship_type')
                if ind>=len(edge):
                    continue
                if edge[ind] == "PARTSToFACILITY":
                    if edge[self.get_edge_index("PARTSToFACILITY", 'target')] in facility_external_demand:
                        src = edge[self.get_edge_index("PARTSToFACILITY", 'source')]
                        if src in raw_demand:
                            raw_demand[src] += edge[self.get_edge_index("PARTSToFACILITY", 'quantity')] * facility_external_demand[edge[self.get_edge_index("PARTSToFACILITY", 'target')]]
                        else:
                            raw_demand[src] = edge[self.get_edge_index("PARTSToFACILITY", 'quantity')] * facility_external_demand[edge[self.get_edge_index("PARTSToFACILITY", 'target')]]
                    else:
                        src = edge[self.get_edge_index("PARTSToFACILITY", 'source')]
                        if src not in list_pop and src not in sa_demand:
                            list_pop.append(edge[self.get_edge_index("PARTSToFACILITY", 'source')])
    
            demand={}
            demand_facility={}
            for key, val in sa_demand.items():
                demand[key] = val[0]
            for key, val in raw_demand.items():
                demand[key] = val

            for key, val in facility_demand.items():
                demand_facility[key] = val
            for key, val in facility_external_demand.items():
                demand_facility[key] = val
            
            self.demand = demand
            self.demand_facility = demand_facility
            self.po_demand = po_demand
            self.bottle_neck = bottle_neck
            self.list_pop = list_pop
            self.facility_supply = supply
    



    def parse_json_to_graph(self, data, timestamp, regression, multistep, task, q=None):
        hetero_data = HeteroData()
        graph_nx = nx.DiGraph() if data['directed'] else nx.Graph()
        
        self.id_map = {}
        idType_map={}

        node_features = {}
        label = {}
        edge_features = {}
        edge_index = {}
        allowed_nodes=[]
        if q is not None and task=='df':
            allowed_nodes = self.part_split_dict[q]
            for node_type, attributes in data['node_types'].items():
                if node_type=='PARTS':
                    continue
                for node in data['node_values'][node_type]:
                    node_id = node[self.get_node_index(node_type, "id")]
                    allowed_nodes.append(node_id)
        else:
            for node_type, attributes in data['node_types'].items():
                for node in data['node_values'][node_type]:
                    node_id = node[self.get_node_index(node_type, "id")]
                    allowed_nodes.append(node_id)

        # Add nodes
        for node_type, attributes in data['node_types'].items():
            if node_type not in self.id_map:
                self.id_map[node_type] = {}
            c = 0  # Counter for node id
            n_id = []
            n_feat = []
            y = []

            
            for node in data['node_values'][node_type]:
                node_id = node[self.get_node_index(node_type, "id")]
                if node_id not in allowed_nodes:
                    self.list_pop.append(node_id)
                    continue
                if node_id in self.list_pop:
                    continue
                attr_dict = dict(zip(attributes, node))
                graph_nx.add_node(node_id, **attr_dict)

                trim_attributes = [item for item in attributes if item in self.get_node_attr(node_type)]

                n_id.append(c)
                self.id_map[node_type][node_id] = c
                idType_map[node_id]=node_type
                c += 1

                n_feat.append([attr_dict[i] if isinstance(attr_dict[i], (int, float)) else self.encode_categorical_value(node_type+'_'+i, attr_dict[i]) for i in trim_attributes])

                if task=='df':
                    if node_type == 'PARTS':
                        if regression:
                            y.append(self.demand[node_id])
                        else:
                            y.append(self.get_y(self.demand[node_id]))
                elif task=='bd':
                    if node_type == 'FACILITY':
                        y.append(self.demand_facility[node_id])
            
            hetero_data[node_type].node_id = n_id
            hetero_data[node_type].x = torch.tensor(n_feat, dtype=torch.float)
            if task=='df':
                if node_type == 'PARTS':
                    if regression:
                        hetero_data[node_type].y = torch.tensor(y, dtype=torch.float)
                    else:
                        hetero_data[node_type].y = torch.tensor(y, dtype=torch.long)
            elif task=='bd':
                if node_type == 'FACILITY':
                    hetero_data[node_type].y = torch.tensor(y, dtype=torch.float)

            node_features[node_type] = torch.tensor(n_feat, dtype=torch.float)
            label[node_type] = torch.tensor(y, dtype=torch.long)

        # Add edges
        for edge_type, attributes in data['relationship_types'].items():
            src = []
            dst = []

            edge_feat = []
            for edge in data['link_values']:
                ind = self.get_edge_index(edge_type, "relationship_type")
                if ind >= len(edge):
                    continue
                if edge[ind] == edge_type:
                    source = edge[self.get_edge_index(edge_type, "source")]
                    target = edge[self.get_edge_index(edge_type, "target")]

                    if source in self.list_pop or target in self.list_pop:
                        continue

                    
                    edge_attr = dict(zip(attributes, edge[:]))
                    graph_nx.add_edge(source, target, **edge_attr, type=edge_type)

                    trim_attributes = [item for item in edge_attr if item in self.get_edge_attr(edge_type)]

                    src.append(self.id_map[idType_map[source]][source])
                    dst.append(self.id_map[idType_map[target]][target])
                    edge_feat.append([edge_attr[i] for i in trim_attributes])

            edge_split = re.split(r'To', edge_type)
            if(not len(edge_split)==1):
                hetero_data[edge_split[0], 'To', edge_split[1]].edge_index = torch.stack((torch.tensor(src), torch.tensor(dst)))
                hetero_data[edge_split[0], 'To', edge_split[1]].edge_attr = torch.tensor(edge_feat, dtype=torch.float)

            edge_index[edge_type] = torch.stack((torch.tensor(src), torch.tensor(dst)))
            edge_features[edge_type] = torch.tensor(edge_feat, dtype=torch.float)

        self.node_features[timestamp] = node_features
        self.edge_features[timestamp] = edge_features
        self.edge_indices[timestamp] = edge_index
        self.node_label[timestamp] = label

        return graph_nx, hetero_data

    def preprocess(self,data):
        lis=[]
        for key, attr in data["link_values"].items():
            lis.extend(attr)
        data["link_values"] = lis

        return data


    def set_dict_values(self, g):
        g_dict = g.to_dict()
        x_dict={}
        edge_index_dict={}
        edge_attr = {}

        for i in g_dict:
            if isinstance(i, tuple):
                edge_index_dict[i] = g_dict[i]["edge_index"]
                if 'edge_attr' in g_dict[i]:
                    edge_attr[i] = g_dict[i]["edge_attr"]
            elif g_dict[i]!={}:
                x_dict[i] = g_dict[i]["x"]
        g.x_dict = x_dict
        g.edge_index_dict = edge_index_dict
        g.edge_attr = edge_attr

        return g

    def set_mask(self, g, type):
        num_nodes = g[type].x.size(0)

        train_ratio = 0.7
        val_ratio = 0.1
        test_ratio = 0.2 
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."
        

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        

        indices = torch.randperm(num_nodes)
        train_count = int(train_ratio * num_nodes)
        val_count = int(val_ratio * num_nodes)
        
        train_indices = indices[:train_count]
        val_indices = indices[train_count:train_count + val_count]
        test_indices = indices[train_count + val_count:]
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        g[type].train_mask = train_mask
        g[type].val_mask = val_mask
        g[type].test_mask = test_mask
    
        return g


    def randomize_values(self, d):
        for key in d:
            d[key] = random.randint(0, 2000)
        return d

    def segregate_by_quantiles(self, d, num_quartiles=4):
        keys = list(d.keys())
        values = np.array([d[k] for k in keys])
        percentiles = [(100 / num_quartiles) * i for i in range(1, num_quartiles)]
        cuts = np.percentile(values, percentiles)
        qdict = {f"q{i}": [] for i in range(1, num_quartiles + 1)}
        for k in keys:
            v = d[k]
            quantile_index = 1  # Start from the first quantile
            for cutoff in cuts:
                if v <= cutoff:
                    break
                quantile_index += 1
            qdict[f"q{quantile_index}"].append(k)
        return qdict

    def sparsity_dict(self, data, timestamp):
        for edge in data['link_values']:
            if 'PARTSToFACILITY' in edge:
                quantity = edge[self.get_edge_index('PARTSToFACILITY', 'quantity')]
                src = edge[self.get_edge_index('PARTSToFACILITY', 'source')]
                tgt = edge[self.get_edge_index('PARTSToFACILITY', 'target')]
                if src in self.parts_dict:
                    if tgt in self.parts_dict[src]:
                        self.parts_dict[src][tgt].append(quantity)
                    else:
                         self.parts_dict[src][tgt] = [quantity]
                else:
                    self.parts_dict[src] = {tgt:[quantity]}
        parts_variation={}
        for part in self.parts_dict:
            total = 0
            
            for tgt in self.parts_dict[part]:
                nums = self.parts_dict[part][tgt]
                for i in range(1, len(nums)):
                    total += abs(nums[i] - nums[i-1])
                    
            parts_variation[part] = int(total)
            
        parts_variation = self.randomize_values(parts_variation)
        self.part_split_dict = self.segregate_by_quantiles(parts_variation, self.num_quartiles)     

    def determine_edge_type(self, source, target):
        """
        Determines the edge type based on the prefixes of source and target IDs.
        
        Parameters:
            source (str): The source ID of the edge.
            target (str): The target ID of the edge.
        
        Returns:
            str: The edge type in the format '<SourceType>To<TargetType>'.
        """
        # Prefix mapping for node types
        prefix_to_type = {
            "BG_": "BUSINESS_GROUP",
            "PO_": "PRODUCT_OFFERING",
            "P_": "PARTS",
            "W_": "WAREHOUSE",
            "PF_": "PRODUCT_FAMILY",
            "S_": "SUPPLIERS",
            "F_":"FACILITY"
        }
        
        # Identify the source and target types based on prefixes
        source_type = next((v for k, v in prefix_to_type.items() if source.startswith(k)), "UNKNOWN")
        target_type = next((v for k, v in prefix_to_type.items() if target.startswith(k)), "UNKNOWN")
        
        # Construct the edge type
        return f"{source_type}To{target_type}"


    def construct_json_dynamic_format(self, business_group_df, facilities_df, parts_df,
                                      product_families_df, product_offerings_df, 
                                      suppliers_df, warehouses_df, edges_df):

        edges_df = edges_df.drop(columns=['source_type', 'target_type'], errors='ignore')
        edges_df = edges_df.rename(columns={'source_id': 'source', 'target_id': 'target'})

        edge_types = edges_df['relationship_type'].unique() if 'relationship_type' in edges_df.columns else ['default']
        
        relationship_types = {}
        link_values = {}
        
        for edge_type in edge_types:
            edge_data = edges_df[edges_df['relationship_type'] == edge_type] if edge_type != 'default' else edges_df
            
            relevant_columns = edge_data.dropna(axis=1, how='all').columns.tolist()

            relationship_types[edge_type] = relevant_columns
            link_values[edge_type] = edge_data[relevant_columns].values.tolist()
        
        return {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "node_types": {
                "BUSINESS_GROUP": list(business_group_df.columns),
                "FACILITY": list(facilities_df.columns),
                "PARTS": list(parts_df.columns),
                "PRODUCT_FAMILY": list(product_families_df.columns),
                "PRODUCT_OFFERING": list(product_offerings_df.columns),
                "SUPPLIERS": list(suppliers_df.columns),
                "WAREHOUSE": list(warehouses_df.columns)
            },
            "relationship_types": relationship_types,
            "node_values": {
                "BUSINESS_GROUP": business_group_df.values.tolist(),
                "FACILITY": facilities_df.values.tolist(),
                "PARTS": parts_df.values.tolist(),
                "PRODUCT_FAMILY": product_families_df.values.tolist(),
                "PRODUCT_OFFERING": product_offerings_df.values.tolist(),
                "SUPPLIERS": suppliers_df.values.tolist(),
                "WAREHOUSE": warehouses_df.values.tolist()
            },
            "link_values": link_values
        }
        
    def validate_parser(self):
        test_path = self.local_dir+ self.version
        if os.path.exists(test_path):
            json_files = glob.glob(os.path.join(test_path, '*.json'))
            if json_files:
                for file in json_files:
                    os.remove(file)
        c = 0
        
        for folder in os.listdir(test_path):
            path = os.path.join(test_path, folder)
            
            if not os.path.isdir(path) or folder.startswith('.') or folder=='test_json':
                continue
            
            c += 1
    
            business_group_path = os.path.join(path, 'business_group.csv')
            edges_path = os.path.join(path, 'edges.csv')
            facilities_path = os.path.join(path, 'facilities.csv')
            metadata_path = os.path.join(path, 'metadata.csv')
            parts_path = os.path.join(path, 'parts.csv')
            product_families_path = os.path.join(path, 'product_families.csv')
            product_offerings_path = os.path.join(path, 'product_offerings.csv')
            suppliers_path = os.path.join(path, 'suppliers.csv')
            warehouses_path = os.path.join(path, 'warehouses.csv')
    
            business_group_df = pd.read_csv(business_group_path)
            edges_df = pd.read_csv(edges_path)
            facilities_df = pd.read_csv(facilities_path)
            metadata_df = pd.read_csv(metadata_path)
            parts_df = pd.read_csv(parts_path)
            product_families_df = pd.read_csv(product_families_path)
            product_offerings_df = pd.read_csv(product_offerings_path)
            suppliers_df = pd.read_csv(suppliers_path)
            warehouses_df = pd.read_csv(warehouses_path)
    
            edges_df['relationship_type'] = edges_df.apply(
                lambda row: self.determine_edge_type(row['source_id'], row['target_id']), axis=1
            )
    
            structured_json_dynamic_corrected = self.construct_json_dynamic_format(
                business_group_df, facilities_df, parts_df, product_families_df,
                product_offerings_df, suppliers_df, warehouses_df, edges_df
            )
    
            test_json_path = os.path.join(test_path)
            os.makedirs(test_json_path, exist_ok=True)
    
            output_path = os.path.join(test_json_path, str(c) + '.json')
            with open(output_path, 'w') as json_file:
                json.dump(structured_json_dynamic_corrected, json_file, indent=4)


    def add_dummy_edge(self, g, src, rel, dst):
        n_src = len(g[src]['x']) # node_id: [0..n_src]
        n_dst = len(g[dst]['x']) # node_id: [0..n_dst]

        unconnected_dst_nodes = set(range(n_dst))
        
        additional_src_nodes = []
        additional_dst_nodes = []
        
        for unconnected_dst in unconnected_dst_nodes:
            random_src = torch.randint(0, n_src, (1,)).item()  # Randomly select a source node
            additional_src_nodes.append(random_src)
            additional_dst_nodes.append(unconnected_dst)
            
        if additional_src_nodes and additional_dst_nodes:
            new_edges = torch.tensor([additional_src_nodes, additional_dst_nodes])
            g[src, rel, dst].edge_index = new_edges
            
            num_new_edges = new_edges.shape[1]
            new_edge_attr = torch.zeros((num_new_edges, 1))
            g[src, rel, dst].edge_attr = new_edge_attr

        return g

    def create_classes(self, num_classes, demand_values):
        demand_values = np.array(list(demand_values.values()))
        percentiles = np.linspace(0, 100, num_classes + 1)
        class_boundaries = np.percentile(demand_values, percentiles)
        self.class_ranges = {i: (class_boundaries[i], class_boundaries[i + 1]) 
                           for i in range(num_classes)}
    

    def get_y(self, value):
        for class_index, (lower_bound, upper_bound) in self.class_ranges.items():
            if lower_bound < value <= upper_bound:
                return class_index
        return len(self.class_ranges) - 1

    def create_temporal_graph(self,regression=False, multistep = True, out_steps = 2, task = 'df', threshold=10, q=None):
        self.fetch_timestamps()

        temporal_graphs = {}
        first_timestamp = True
        json={}
        demand={}
        
        for timestamp in self.timestamps:
            data = self.fetch_json(timestamp)
            
            self.node_schema = data["node_types"]
            self.edge_schema = data["relationship_types"]
        
            self.set_date_df(timestamp, data)
            json[timestamp] = data
            
            
            data = self.preprocess(data)
            self.compute_demand(data, threshold)
            demand[timestamp] = self.demand
            
            if first_timestamp:
                self.create_classes(self.num_classes, self.demand)
                first_timestamp = False
                self.sparsity_dict(data, timestamp)
                
            graph_nx, hetero_data = self.parse_json_to_graph(data, timestamp, regression, multistep, task, q)
            
            hetero_data = self.add_dummy_edge(hetero_data, "WAREHOUSE", "To", "SUPPLIERS")
            hetero_data = self.add_dummy_edge(hetero_data, "PRODUCT_FAMILY", "To", "BUSINESS_GROUP")
            
            hetero_data = self.set_dict_values(hetero_data)

            if regression:
                hetero_data.num_classes = 1
            else:
                hetero_data.num_classes = self.num_classes

            if task =='df':
                hetero_data = self.set_mask(hetero_data, 'PARTS')
            else:
                hetero_data = self.set_mask(hetero_data, 'FACILITY')
            temporal_graphs[timestamp] = (graph_nx, hetero_data)

            
            self.set_df(timestamp)
            self.set_po_df(timestamp)

        self.set_extended_df(json, 'PARTS')
        hetero_list = copy.deepcopy(temporal_graphs)
        if multistep:
            hetero_obj = self.multistep_data(hetero_list, out_steps)
        else:
            hetero_obj = temporal_graphs
        return temporal_graphs, hetero_obj
