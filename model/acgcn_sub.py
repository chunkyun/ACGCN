import torch
import torch.nn as nn
from utils.GCNPredictor import GCNPredictor


class ACGCN_SUB(nn.Module):

    def __init__(self, args):
        super(ACGCN_SUB, self).__init__()
        
        number_atom_features = 32
        graph_conv_layers_core: list = [256, 256]
        graph_conv_layers_sub: list = [64, 64]
        activation = None
        residual: bool = True
        batchnorm: bool = True
        dropout: float = args['DROP_OUT']
        out_size_core = 256
        out_size_sub = 64
        predictor_hidden_feats_core: int = 256
        predictor_hidden_feats_sub: int = 64
        predictor_dropout: float = args['DROP_OUT']
        num_gnn_layers = len(graph_conv_layers_core)

        self.gcn_layer_core = GCNPredictor(
                            in_feats=number_atom_features,
                            hidden_feats=graph_conv_layers_core,
                            activation=activation,
                            residual=[residual] * num_gnn_layers,
                            batchnorm=[batchnorm] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            n_tasks=out_size_core,
                            predictor_hidden_feats=predictor_hidden_feats_core,
                            predictor_dropout=predictor_dropout)
        
        self.gcn_layer_sub = GCNPredictor(
                            in_feats=number_atom_features,
                            hidden_feats=graph_conv_layers_sub,
                            activation=activation,
                            residual=[residual] * num_gnn_layers,
                            batchnorm=[batchnorm] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            n_tasks=out_size_sub,
                            predictor_hidden_feats=predictor_hidden_feats_sub,
                            predictor_dropout=predictor_dropout)
        
        self.fc_layer2 = nn.Sequential(nn.Linear((256+64*2), (256+64*2)*2),
                           nn.BatchNorm1d((256+64*2)*2),
                           nn.ReLU(inplace=True),
                           nn.Dropout(args['DROP_OUT']))

        self.out_layer = nn.Linear((256+64*2)*2, 1)

    def forward(self, batch_core, batch_sub1, batch_sub2):
        
        core = self.gcn_layer_core(batch_core, batch_core.ndata['x'].float())
        sub1 = self.gcn_layer_sub(batch_sub1, batch_sub1.ndata['x'].float())
        sub2 = self.gcn_layer_sub(batch_sub2, batch_sub2.ndata['x'].float())
        
        out = torch.cat((core, sub1, sub2), axis=1)
        out = self.fc_layer2(out)
        out = self.out_layer(out)
        out = torch.sigmoid(out)
        out = out.squeeze(-1)

        return out
