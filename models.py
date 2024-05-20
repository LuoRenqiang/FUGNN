import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 
class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)
        # self.eig_w = nn.Linear(1, hidden_dim)

    def forward(self, eignvalue):

        eignvalue_con = eignvalue * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(eignvalue.device)
        position = eignvalue_con.unsqueeze(1) * div
        eignvalue_pos = torch.cat((eignvalue.unsqueeze(1), torch.sin(position), torch.cos(position)), dim=1)
        # eignvalue_pos = eignvalue.unsqueeze(1) # [4,1]
        return self.eig_w(eignvalue_pos)

# mlp
class FeedForward(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, feature):
        feature = self.layer1(feature)
        feature = self.gelu(feature)
        feature = self.layer2(feature)
        return feature


class FULayer(nn.Module):

    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm='none'):
        super(FULayer, self).__init__()
        # self.nheads+1, hidden_dim
        self.prop_dropout = nn.Dropout(prop_dropout)

        if norm == 'none':
            self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        else:
            self.weight = nn.Parameter(torch.empty((1, nbases, ncombines)))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        if norm == 'layer':
            self.norm = nn.LayerNorm(ncombines)
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(ncombines)
        else:
            self.norm = None

    def forward(self, feature):
        # print('feature.shape',feature.shape) # [67796, 5, 128]
        # print('self.weight',self.weight.shape) # [1, 2, 128]
        feature = self.prop_dropout(feature) * self.weight
        feature = torch.sum(feature, dim=1)

        if self.norm is not None:
            feature = self.norm(feature)
            feature = F.relu(feature)

        return feature


class FUGNN(nn.Module):

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, nheads=1,
                tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):
        super(FUGNN, self).__init__()

        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        # feat_encoder = mlp (with relu)
        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )

        # linear_encoder = onne layer Linear
        self.linear_encoder = nn.Linear(nfeat, hidden_dim)
        
        self.classify = nn.Linear(hidden_dim, nclass)

        # position encoding
        self.eignvalue_encoder = SineEncoding(hidden_dim)
        
        # linear decoder
        self.decoder = nn.Linear(hidden_dim, nheads)

        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.oc_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.oc_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        
        # mlp with gelu
        self.oc = FeedForward(hidden_dim, hidden_dim, hidden_dim)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)
        if norm == 'none': # nlayer
            # self.layers = nn.ModuleList([FULayer(2, nclass, prop_dropout, norm=norm) for i in range(nlayer)])
            self.layers = nn.ModuleList([FULayer(self.nheads+1, nclass, prop_dropout, norm=norm) for i in range(nlayer)])
        else:
            # self.layers = nn.ModuleList([FULayer(2, hidden_dim, prop_dropout, norm=norm) for i in range(nlayer)])
            self.layers = nn.ModuleList([FULayer(self.nheads+1, hidden_dim, prop_dropout, norm=norm) for i in range(nlayer)])

    def forward(self, eignvalue, eignvector, feature):
        eignvector_T = eignvector.permute(1, 0)
        if self.norm == 'none':
            h = self.feat_dp1(feature)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(feature)
            h = self.linear_encoder(h) # only this
            
        # position encoding - no h 
        eig = self.eignvalue_encoder(eignvalue) 
        # eig = self.eignvalue_encoder(eignvector) 

        # multi head attention with residual connection - no h
        mha_eig = self.mha_norm(eig)
        mha_eig, _ = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + self.mha_dropout(mha_eig)

        # mlp with gelu - no h
        oc_eig = self.oc_norm(eig)
        oc_eig = self.oc(oc_eig)
        # eig = self.oc_dropout(oc_eig)
        eig = self.oc_dropout(oc_eig)+ eig

        # eig_faxgnn = eig # 
        eig_faxgnn = self.decoder(eig)
        
        for conv in self.layers:
            basic_feats = [h]
            eignvector_conv = eignvector_T @ h # 
            for i in range(self.nheads):
                basic_feats.append(eignvector @ (eig_faxgnn[:, i].unsqueeze(1) * eignvector_conv)) 
            basic_feats = torch.stack(basic_feats, axis=1)
            h = conv(basic_feats)

        if self.norm == 'none':
            return h
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h


