import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder,StandardScaler

companies = pd.read_csv('/home/jovyan/hfactory_magic_folders/financial_graph_mining_for_customers___supply_chains_assessment/static_data_all_x.csv',delimiter = ";",header = 0)
transactions = pd.read_csv('/home/jovyan/hfactory_magic_folders/financial_graph_mining_for_customers___supply_chains_assessment/transactions_x.csv',delimiter = ";",header = 0)

df4 = companies[companies['QUARTER'] == 'q4'][['ID' ,'REGION',  'ESG_SCORE', 'CATEGORY'  ,'T_LOCAL_MT_ACTIF_SOCIAL', 'T_LOCAL_TX_PD']]

node_ids = df4.iloc[:, 0]  # Assuming the first column contains the node IDs
X = df4.iloc[:, 1:5]  # Assuming the remaining columns contain the node features
node_labels = df4.iloc[:,[0,5]]

#label_encoder = LabelEncoder()
node_labels['T_LOCAL_TX_PD'] = label_encoder.fit_transform(node_labels['T_LOCAL_TX_PD'])

label_encoder_X = LabelEncoder()
X.iloc[:, 0] = label_encoder_X.fit_transform(X.iloc[:, 0])

num_classes = len(label_encoder_X.classes_)
print("Number of REGIONS:", num_classes)

label_encoder_X = LabelEncoder()
X.iloc[:, 2] = label_encoder_X.fit_transform(X.iloc[:, 2])

num_classes = len(label_encoder_X.classes_)
print("Number of CATEGORIES:", num_classes)

num_classes = len(label_encoder.classes_)
print("Number of CLASSES:", num_classes)

columns_to_encode = [0, 2]

# Extract the columns to be one-hot encoded
columns_subset = X.iloc[:, columns_to_encode]

# Perform one-hot encoding on the subset of columns
encoded_columns = pd.get_dummies(columns_subset, drop_first=True)  # Set drop_first=True to avoid multicollinearity

# Drop the original columns from the DataFrame
df = X.drop(columns_subset.columns, axis=1)

# Concatenate the original DataFrame with the encoded columns
df = pd.concat([df, encoded_columns], axis=1)

df['log_actif_social'] = np.log(df['T_LOCAL_MT_ACTIF_SOCIAL'])

scaler = StandardScaler()

# Fit the scaler to your data
scaler.fit(df)

# Transform the data (perform standard scaling)
df = pd.DataFrame(scaler.transform(df), columns=df.columns)

print(node_ids)
print(df)
print(node_labels)

# Step 3: Create a graph object
G = nx.Graph()

# Step 4: Add nodes to the graph and set their attributes
for node_id, features in zip(node_ids, df.values):
    G.add_node(node_id, features=features.tolist())

print(len(G))

transactions['log_column'] = np.log(transactions['TX_AMOUNT'])
value_counts = transactions.groupby(['ID', 'COUNTERPARTY']).size()
mean_log = transactions.groupby(['ID', 'COUNTERPARTY'])['log_column'].mean()

all_needed = pd.DataFrame()
all_needed['value_counts'] = value_counts
all_needed['mean_log'] = mean_log

columns_to_scale = ['value_counts', 'mean_log']

# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler to the selected columns
scaler.fit(all_needed[columns_to_scale])

# Transform the selected columns (perform standard scaling)
all_needed[columns_to_scale] = scaler.transform(all_needed[columns_to_scale])

print(all_needed)

for (node1, node2), edge_data in all_needed.iterrows():
    # Extract edge features
    edge_features = edge_data.to_dict()
    
    # Add edge to the graph with features
    G.add_edge(node1, node2, **edge_features)

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
average_degree = sum(dict(G.degree()).values()) / num_nodes

# Print the statistics
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)
print("Average degree:", average_degree)

import random

# Assuming G is your NetworkX graph initialized earlier

# Step 1: Select a random node
random_node = random.choice(list(G.nodes()))

# Step 2: Retrieve the features for the random node
node_features = G.nodes[random_node]['features']

print("Features for node", random_node, ":", node_features)

import sys

# Your script code here

# Exit the script
#sys.exit()

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split

# Assuming G is your NetworkX graph initialized earlier
# Assuming node_labels is your DataFrame containing node IDs and corresponding labels

# Step 1: Split the Dataset
train_ids, test_ids = train_test_split(node_labels['ID'], test_size=0.2, random_state=42)
train_labels = node_labels[node_labels['ID'].isin(train_ids)]['T_LOCAL_TX_PD']
test_labels = node_labels[node_labels['ID'].isin(test_ids)]['T_LOCAL_TX_PD']

# Step 2: Prepare the Data
nodes = list(G.nodes())
list_eq = [0]*len(nodes)
dict_eq = {}
pointer = 0
for n in nodes:
    list_eq[pointer] = n
    dict_eq[n] = pointer
    pointer += 1

EQ = np.arange(0,len(nodes))

labels_dict = dict(zip(node_labels['ID'], node_labels['T_LOCAL_TX_PD']))
labels = torch.tensor([labels_dict[node] for node in nodes], dtype=torch.long)

# Extract features for nodes in the graph (assuming each node has 'features' attribute)
# Ensure indices are within the bounds of the number of nodes
x = torch.tensor([G.nodes[node]['features'] for node in nodes], dtype=torch.float)

# Extract edges from the graph and ensure the indices are within the bounds of the number of nodes
edges = [(dict_eq[edge[0]], dict_eq[edge[1]]) for edge in G.edges()]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# If you have edge features, extract them in a similar manner
# edge_features = torch.tensor([G.edges[edge]['edge_features'] for edge in G.edges()], dtype=torch.float)

train_mask = torch.tensor([node in train_ids.tolist() for node in nodes], dtype=torch.bool)
test_mask = ~train_mask  # Invert train_mask to get test_mask

print(train_mask)
print(test_mask)
print(train_mask.shape)
print(torch.sum(train_mask).item())

data = Data(x=x, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask = test_mask)

# Step 3: Define the Model
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, num_classes)  # Define num_classes as before
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Step 4: Train the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
class_weights = torch.tensor([1.0 / count for count in torch.bincount(data.y[train_mask])], dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

weight_decay = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

for epoch in range(1000):
    train()


from sklearn.metrics import balanced_accuracy_score

# Step 5: Evaluate the Model
model.eval()
with torch.no_grad():
    _, pred = model(data).max(dim=1)
    correct = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
    accuracy = correct / data.train_mask.sum().item()
    
    # Compute balanced accuracy
    balanced_acc = balanced_accuracy_score(data.y[data.train_mask].cpu(), pred[data.train_mask].cpu())
    
    print('Accuracy: {:.4f}'.format(accuracy))
    print('Balanced Accuracy: {:.4f}'.format(balanced_acc))

# Assuming model, data, and test_mask are defined as before

model.eval()
with torch.no_grad():
    _, pred = model(data).max(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    accuracy = correct / data.test_mask.sum().item()
    
    # Compute balanced accuracy
    balanced_acc = balanced_accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
    
    print('Accuracy: {:.4f}'.format(accuracy))
    print('Balanced Accuracy: {:.4f}'.format(balanced_acc))