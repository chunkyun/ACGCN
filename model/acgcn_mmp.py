import torch
import torch.nn as nn
from utils.GCNPredictor import GCNPredictor


class ACGCN_MMP(nn.Module):

    def __init__(self, args):
        super(ACGCN_MMP, self).__init__()

        number_atom_features = 32
        graph_conv_layers: list = [128, 128]
        activation = None
        residual: bool = True
        batchnorm: bool = True
        dropout: float = args['DROP_OUT']#0.5 ###
        out_size = 128*2
        predictor_hidden_feats: int = 128*2
        predictor_dropout: float = args['DROP_OUT'] ###
        num_gnn_layers = len(graph_conv_layers)

        self.gcn_layer = GCNPredictor(in_feats=number_atom_features,
                                    hidden_feats=graph_conv_layers,
                                    activation=activation,
                                    residual=[residual] * num_gnn_layers,
                                    batchnorm=[batchnorm] * num_gnn_layers,
                                    dropout=[dropout] * num_gnn_layers,
                                    n_tasks=out_size,
                                    predictor_hidden_feats=predictor_hidden_feats,
                                    predictor_dropout=predictor_dropout)

        self.fc_layer2 = nn.Sequential(nn.Linear(128*4, 128*8),
                                           nn.BatchNorm1d(128*8),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout(args['DROP_OUT'])) ###

        self.out_layer = nn.Linear(128*8, 1)

    def forward(self, batch_smiles1, batch_smiles2):
        smiles1 = self.gcn_layer(batch_smiles1, batch_smiles1.ndata['x'].float())
        smiles2 = self.gcn_layer(batch_smiles2, batch_smiles2.ndata['x'].float())
        
        out = torch.cat((smiles1, smiles2), axis=1)
        out = self.fc_layer2(out)
        out = self.out_layer(out)
        out = torch.sigmoid(out)
        out = out.squeeze(-1)
        
        return out