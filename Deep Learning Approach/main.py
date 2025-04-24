# My final approach uses Graph-based assembly sequence prediction using PyTorch Geometric (Graph Neural Network).

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

# 1. Create a toy dataset of LEGO-like structures represented as graphs.
# Each graph's nodes are bricks, and edges indicate connections (attachments) between bricks.
# We also define a target assembly order for each graph (as a list or tensor of step indices for each node).

dataset = []

#Graph 1: Two bricks connected (a simple pair).
edge_index = torch.tensor([[0], [1]], dtype=torch.long)  
edge_index = torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1)  
num_nodes = 2
x = torch.ones((num_nodes, 1))  
target_steps = torch.tensor([0.0, 1.0], dtype=torch.float)  
data1 = Data(x=x, edge_index=edge_index, y=target_steps)

dataset.append(data1)

#Graph 2: Three bricks in a linear chain (like a tower of 3 bricks).
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  
edge_index = torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1)  
num_nodes = 3
x = torch.ones((num_nodes, 1))
target_steps = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float)
data2 = Data(x=x, edge_index=edge_index, y=target_steps)

dataset.append(data2)

#Graph 3: Three bricks in a star configuration (brick 0 connected to brick 1 and 2).
edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)  # edges: 0-1, 0-2
edge_index = torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1)  # add 1-0, 2-0
num_nodes = 3
x = torch.ones((num_nodes, 1))
target_steps = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float)
data3 = Data(x=x, edge_index=edge_index, y=target_steps)

dataset.append(data3)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

#Graph Neural Network model class
class AssemblyGNN(nn.Module):
    # initializing the covolutional layers
    def __init__(self, in_channels, hidden_channels):
        super(AssemblyGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out_lin = nn.Linear(hidden_channels, 1)
    
    # Forward Pass
    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        out = self.out_lin(h)
        return out.view(-1)

# Initialize model, loss, optimizer
model = AssemblyGNN(in_channels=1, hidden_channels=16)  
criterion = nn.MSELoss()   
optimizer = optim.Adam(model.parameters(), lr=0.01)

model.train()
epochs = 100
for epoch in range(epochs):
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

model.eval()

# This function when given a lego structure graph will predict an assembly sequence as an ordering of node indices
def predict_assembly_order(graph_data):
    with torch.no_grad():
        scores = model(graph_data.x, graph_data.edge_index, torch.zeros(graph_data.num_nodes, dtype=torch.long))
        scores = scores.cpu().numpy()
    order = list(np.argsort(scores))
    return order, scores

test_graph = data2 
pred_order, pred_scores = predict_assembly_order(test_graph)
print("\nTest on a 3-brick tower structure (Graph 2):")
print("Predicted assembly scores per node:", pred_scores)
print("Predicted assembly order (node indices):", pred_order)
print("Actual (target) assembly order we trained on: [0, 1, 2]")  


edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
edge_index = torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1)
num_nodes = 4
x = torch.ones((num_nodes, 1))
new_graph = Data(x=x, edge_index=edge_index)
pred_order, pred_scores = predict_assembly_order(new_graph)
print("\nTest on a new 4-brick chain structure (not seen in training):")
print("Predicted assembly scores per node:", pred_scores)
print("Predicted assembly order (node indices):", pred_order)
