import streamlit as st
import urllib3
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

class TemporalHeterogeneousGraphParser:
    def __init__(self, base_url, version, headers,meta_data_path, use_local_files=False, local_dir=None,  num_classes =20):
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
        self.df_date = None
        self.df_dict={}
        
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
                
    def set_extended_df(self, data_dict, type):
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
            e_type = edge[0]
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
            if edge[self.get_edge_index("FACILITYToPRODUCT_OFFERING", 'relationship_type')] == "FACILITYToPRODUCT_OFFERING":
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
                facility_demand[id] = dem
                if po_demand[target][0]/po_demand[target][2] > dem + threshold:
                    bottle_neck[id] = 1
                else:
                    bottle_neck[id] = 0

        for edge in data['link_values']:
            if edge[self.get_edge_index("PARTSToFACILITY", 'relationship_type')] == "PARTSToFACILITY":
                if edge[self.get_edge_index("PARTSToFACILITY", 'target')] in facility_demand:
                    src = edge[self.get_edge_index("PARTSToFACILITY", 'source')]
                    if src in sa_demand:
                        sa_demand[src] += [edge[self.get_edge_index("PARTSToFACILITY", 'quantity')] * facility_demand[edge[self.get_edge_index("PARTSToFACILITY", 'target')]], 0, 0]
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
                    if sa_demand[target][0]/  sa_demand[target][2] > dem + threshold:
                        bottle_neck[id] = 1
                    else:
                        bottle_neck[id] = 0
        
        for edge in data['link_values']:
            if edge[self.get_edge_index("PARTSToFACILITY", 'relationship_type')] == "PARTSToFACILITY":
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
        for key, val in sa_demand.items():
            demand[key] = val[0]
        for key, val in raw_demand.items():
            demand[key] = val
        
        self.demand = demand
        self.bottle_neck = bottle_neck
        self.list_pop = list_pop



    def parse_json_to_graph(self, data, timestamp, regression, multistep, task):
        hetero_data = HeteroData()
        graph_nx = nx.DiGraph() if data['directed'] else nx.Graph()
        
        self.id_map = {}
        idType_map={}

        node_features = {}
        label = {}
        edge_features = {}
        edge_index = {}

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
                        y.append(self.bottle_neck[node_id])
            
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
                if edge[0] == edge_type:
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

    def create_temporal_graph(self,regression=False, multistep = True, out_steps = 2, task = 'df', threshold=10):
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
            
            graph_nx, hetero_data = self.parse_json_to_graph(data, timestamp, regression, multistep, task)
            
            hetero_data = self.add_dummy_edge(hetero_data, "WAREHOUSE", "To", "SUPPLIERS")
            hetero_data = self.add_dummy_edge(hetero_data, "PRODUCT_FAMILY", "To", "BUSINESS_GROUP")
            
            hetero_data = self.set_dict_values(hetero_data)
            if task =='df':
                hetero_data = self.set_mask(hetero_data, 'PARTS')
            else:
                hetero_data = self.set_mask(hetero_data, 'FACILITY')
            temporal_graphs[timestamp] = (graph_nx, hetero_data)

            self.set_df(timestamp)

        self.set_extended_df(json, 'PARTS')
        hetero_list = copy.deepcopy(temporal_graphs)
        if multistep:
            hetero_obj = self.multistep_data(hetero_list, out_steps)
        else:
            hetero_obj = temporal_graphs
        # st.write(temporal_graphs)
        return temporal_graphs, hetero_obj
        # return 0,1
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
# class Train:
#     def __init__(self, G):
#         """
#         Initializes the Train class with metadata from the input graph.
        
#         Args:
#             G: Input graph object, expected to have a metadata method.
#         """
#         self.metadata = G.metadata()
#         self.model = None
#         self.G = G

#     def set_model(self, hidden_channels=64, out_channels=20, layer_config=None):
#         """
#         Configures the model with the given parameters and converts it to a heterogeneous model.
        
#         Args:
#             hidden_channels: Default number of hidden channels (can be overridden by layer config)
#             out_channels: Number of output channels for the final layer
#             layer_config: Dictionary of layer configurations in the format:
#                 {
#                     'layer1': {
#                         'class': LayerClass,
#                         'params': {
#                             'heads': 4,
#                             'dropout': 0.2,
#                             ...
#                         }
#                     },
#                     'layer2': {
#                         ...
#                     }
#                 }
#         """
#         if layer_config is None or len(layer_config) < 2:
#             raise ValueError("layer_config must contain at least two layers.")
        
#         self.hidden_channels = hidden_channels
#         self.out_channels = out_channels
#         self.layer_config = layer_config

#         class GNN_NodeClassification(torch.nn.Module):
#             def __init__(self, hidden_channels, out_channels, layer_config):
#                 super().__init__()
                
#                 # Dynamically create layers with their specific configurations
#                 self.layers = torch.nn.ModuleList()
#                 layer_keys = list(layer_config.keys())
                
#                 for i, layer_key in enumerate(layer_keys):
#                     layer_spec = layer_config[layer_key]
#                     layer_class = layer_spec['class']
#                     layer_params = layer_spec.get('params', {})
                    
#                     # Set input/output dimensions based on position
#                     if i == 0:  # First layer
#                         layer_params['in_channels'] = (-1, -1)
#                         layer_params['out_channels'] = hidden_channels
#                     elif i == len(layer_keys) - 1:  # Last layer
#                         layer_params['in_channels'] = (-1, -1)
#                         layer_params['out_channels'] = out_channels
#                     else:  # Hidden layers
#                         layer_params['in_channels'] = (-1, -1)
#                         layer_params['out_channels'] = hidden_channels
                    
#                     # Create layer instance with configured parameters
#                     self.layers.append(layer_class(**layer_params))

#             def forward(self, x, edge_index):
#                 for layer in self.layers[:-1]:
#                     # Apply layer with activation
#                     x = layer(x, edge_index)
#                     if isinstance(x, tuple):  # Handle multi-head output (e.g., from GAT)
#                         x = x[0]
#                     x = x.relu()
                
#                 # Final layer without activation
#                 x = self.layers[-1](x, edge_index)
#                 if isinstance(x, tuple):
#                     x = x[0]
#                 return x

#         model = GNN_NodeClassification(hidden_channels, out_channels, layer_config)
#         self.model = to_hetero(model, self.metadata, aggr='sum')
    
#     def train(self, num_epochs, optimizer, loss, label, patience=5):
#         """
#         Trains the model while monitoring validation loss to prevent overfitting.
        
#         Args:
#             num_epochs: Number of epochs to train.
#             optimizer: Optimizer for updating model parameters.
#             loss: Loss function.
#             label: The label node type for which the training mask and target are applied.
#             patience: Number of epochs to wait for validation loss improvement before early stopping.
#         """
#         if self.model is None:
#             raise ValueError("Model has not been set. Use set_model() before training.")
        
#         # Initialize loss tracking
#         self.train_losses = []
#         self.test_losses = []
#         best_test_loss = float('inf')
#         patience_counter = 0
#         best_model_state = None
        
#         for epoch in range(num_epochs):
#             # Training phase
#             self.model.train()
#             optimizer.zero_grad()
            
#             # Forward pass
#             out = self.model(self.G.x_dict, self.G.edge_index_dict)
            
#             # Get training masks and compute predictions
#             train_mask = self.G[label]['train_mask']
#             y_train_true = self.G[label].y[train_mask].cpu().numpy()
#             y_train_pred = out[label][train_mask].argmax(dim=1).cpu().numpy()
            
#             # Compute training loss and backpropagate
#             train_loss = loss(out[label][train_mask], self.G[label].y[train_mask])
#             train_loss.backward()
#             optimizer.step()
            
#             # Calculate training metrics
#             train_accuracy = accuracy_score(y_train_true, y_train_pred)
#             self.train_losses.append(float(train_loss))
            
#             # Testing phase
#             self.model.eval()
#             with torch.no_grad():
#                 # Get test masks and compute predictions
#                 test_mask = self.G[label]['test_mask']
#                 y_test_true = self.G[label].y[test_mask].cpu().numpy()
#                 y_test_pred = out[label][test_mask].argmax(dim=1).cpu().numpy()
                
#                 # Compute test loss and accuracy
#                 test_loss = loss(out[label][test_mask], self.G[label].y[test_mask])
#                 test_accuracy = accuracy_score(y_test_true, y_test_pred)
#                 self.test_losses.append(float(test_loss))
                
#                 # Early stopping check
#                 if test_loss < best_test_loss:
#                     best_test_loss = test_loss
#                     patience_counter = 0
#                     # Save best model state
#                     best_model_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
#                 else:
#                     patience_counter += 1
#                     if patience_counter >= patience:
#                         print(f"Early stopping triggered at epoch {epoch + 1}")
#                         # Restore best model state
#                         self.model.load_state_dict(best_model_state)
#                         break
            
#             # Print epoch metrics
#             print(f"Epoch {epoch + 1}/{num_epochs}")
#             print(f"Train Loss: {float(train_loss):.4f}, Train Accuracy: {train_accuracy:.4f}")
#             print(f"Test Loss: {float(test_loss):.4f}, Test Accuracy: {test_accuracy:.4f}")
#             print("-" * 50)
        
#         # Final model evaluation
#         self.model.eval()
#         with torch.no_grad():
#             final_out = self.model(self.G.x_dict, self.G.edge_index_dict)
#             test_mask = self.G[label]['test_mask']
#             final_test_loss = float(loss(final_out[label][test_mask], self.G[label].y[test_mask]))
#             final_pred = final_out[label][test_mask].argmax(dim=1).cpu().numpy()
#             final_accuracy = accuracy_score(self.G[label].y[test_mask].cpu().numpy(), final_pred)
            
#             print("\nTraining Complete!")
#             print(f"Final Test Loss: {final_test_loss:.4f}")
#             print(f"Final Test Accuracy: {final_accuracy:.4f}")
            
#         return {
#             'train_losses': self.train_losses,
#             'test_losses': self.test_losses,
#             'final_test_loss': final_test_loss,
#             'final_accuracy': final_accuracy,
#             'best_model_state': best_model_state
#         }
#     def test(self, loss, label="PRODUCT_OFFERING"):
#         if self.model is None:
#             raise ValueError("Model has not been set. Use set_model() before testing.")
        
#         self.model.eval()
#         with torch.no_grad():
#             out = self.model(self.G.x_dict, self.G.edge_index_dict)
#             mask = self.G[label]['test_mask']  # Use 'test_mask' for testing
#             y_true = self.G[label].y[mask].cpu().numpy()
#             y_pred = out[label][mask].argmax(dim=1).cpu().numpy()
        
#             test_loss = loss(out[label][mask], self.G[label].y[mask])
        
#             accuracy = accuracy_score(y_true, y_pred)
#             precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
#             recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
#             f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
#             print(f"Test Loss: {test_loss:.4f}, "
#                 f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
#                 f"Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        
#             metrics = {
#                 'test_loss': test_loss.item(),
#                 'accuracy': accuracy,
#                 'precision': precision,
#                 'recall': recall,
#                 'f1': f1
#             }
        
#             return metrics


#     def analytics(self, data_loader):
#         """
#         Evaluates the model on the provided data loader and generates analytics.
        
#         Args:
#             data_loader: A PyTorch DataLoader for loading the evaluation data.
#         """
#         if self.model is None:
#             raise ValueError("Model has not been set. Use set_model() before analytics.")
        
#         self.model.eval()
#         with torch.no_grad():
#             for data in data_loader:
#                 out = self.model(data.x_dict, data.edge_index_dict)
#                 print("Output:", out)
                
                

