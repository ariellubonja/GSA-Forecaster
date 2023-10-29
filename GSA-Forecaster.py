import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool



class GraphSequenceAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphSequenceAttentionLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.att = torch.nn.Parameter(torch.Tensor(1, 2 * out_channels)) # TODO ??

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E] # TODO Isn't this [E, 2]?

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Prepare for attention score calculation and message passing.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 3: Initiate message passing using the transformed features and edge index.
        # The actual attention score calculation occurs during message passing.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention score.
        x_j = x_j.view(-1, self.att.size(1) // 2)
        x_i = x_i.view(-1, self.att.size(1) // 2)

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = torch.nn.functional.leaky_relu(alpha)
        alpha = torch.nn.functional.softmax(alpha, dim=0)

        # Compute updated features and attention scores.
        return x_j * alpha.view(-1, 1)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out



class GSATransformer(nn.Module):
    def __init__(self, num_layers, graph_feat_size, hidden_size, num_heads, dropout_rate):
        super(GSATransformer, self).__init__()

        # Layer list to hold the Graph Sequence Attention layers
        self.gsa_layers = nn.ModuleList()

        # Constructing the GSA layers
        for _ in range(num_layers):
            gsa_layer = GraphSequenceAttentionLayer(graph_feat_size, hidden_size)
            self.gsa_layers.append(gsa_layer)

        # Layer Norm
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Position-wise Feed-Forward Networks
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        # Additional dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Readout layer: Aggregate graph-level features
        self.readout = global_mean_pool

        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)  # Replace with appropriate output size

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Pass through each Graph Sequence Attention layer
        for layer in self.gsa_layers:
            x = layer(x, edge_index)
            x = F.relu(x)  # Apply non-linearity
            x = self.dropout(x)  # Apply dropout

        x = self.layer_norm(x)

        x = self.feed_forward(x) # Position-wise feed-forward networks

        # Aggregate features to get graph-level representation
        x = self.readout(x, data.batch)  # data.batch provides batch-wise graph information

        # Pass through the output layer to get final prediction
        out = self.output_layer(x)

        return out



class TemporalGraphAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(TemporalGraphAttentionLayer, self).__init__(aggr='add')  # "Add" aggregation (sum)
        # ... (initialization remains the same)

    def forward(self, x, edge_index):
        # x shape: [N, T, F] where N is the number of nodes, T is the number of time steps, and F is the number of features

        # Possibly separate the features from different time steps and process them individually
        # or extend the attention mechanism to handle the 3D shape.

        # ... (preprocessing, if needed)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, time_steps=x.size(1))

    def message(self, x_j, time_steps):
        # x_j has shape [E, T, F]

        # You might need to calculate attention scores for each time step, 
        # then combine them in some way (e.g., averaging, a more complex attention mechanism, etc.)

        # This is a placeholder for the actual attention mechanism
        # that you'd need to implement, which might be quite complex
        # and require significant changes to the structure of the layer.
        attention_scores_over_time = self.calculate_attention(x_j)

        return attention_scores_over_time  # Shape might be [E, T, F], or maybe [E, F] if you combine time steps

    # ... (rest of the class)




class TemporalGSATransformer(nn.Module):
    def __init__(self, ...):
        # ... (initialization, creating layers, etc.)

    def forward(self, data):
        x = data.x  # x might have shape [N, T, F]

        # Process the input through each layer, handling the 3D shape appropriately.
        # This might involve changes to the layer structure or the creation of new types of layers.

        # ...

        # You would also need to decide how to aggregate the data across time steps for the output.
        # For example, are you predicting something for each time step, or just for the last time step?
        # This decision would affect the shape of the output and possibly the loss function.

        return output
