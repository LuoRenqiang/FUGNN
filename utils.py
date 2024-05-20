import random
import numpy as np
import os
import torch
import torch.nn as nn
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
import pandas as pd
import scipy.sparse as sp
import time

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False

# inf
def load_dataset(dataset, sens_attr, predict_attr, path, label_number, test_idx=False):
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    print('dataset train test split: ',dataset)
    header = list(idx_features_labels.columns) 
    header.remove(predict_attr) 
    header.remove('user_id')
    # build relationship
    # if os.path.exists(f'{path}/{dataset}_edges.txt'):
    #     edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int') 
    # else:
    #     edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
    #     np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered) 

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32) 
    labels = idx_features_labels[predict_attr].values 
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)} 
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),  
    #                  dtype=int).reshape(edges_unordered.shape)
    # adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)

    # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense())) 
    labels = torch.LongTensor(labels) 
    labels[labels >1] =1
    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]  # 
    label_idx_1 = np.where(labels==1)[0]  # 
    random.shuffle(label_idx_0) 
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], 
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], 
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])

    idx_test=None
    if test_idx:
        # idx_test = label_idx[label_number:]
        idx_test = np.append(label_idx_0[label_number//2:],label_idx_1[label_number//2:])
        idx_val = idx_test
    else:
        # idx_test = label_idx[int(0.75 * len(label_idx)):]
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    # idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
    
    sens = idx_features_labels[sens_attr].values.astype(int) 
    sens = torch.FloatTensor(sens) 
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # check 
    if test_idx:
      save_path='data/inf/{}_information_test_idx.pt'.format(dataset)
    else:
      save_path='data/inf/{}_information.pt'.format(dataset)
    torch.save([features, labels, idx_train, idx_val, idx_test, sens],
                save_path)

def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="data/ori/bail/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    
    # # Normalize School
    # idx_features_labels['SCHOOL'] = 2*(idx_features_labels['SCHOOL']-idx_features_labels['SCHOOL'].min()).div(idx_features_labels['SCHOOL'].max() - idx_features_labels['SCHOOL'].min()) - 1

    # # Normalize RULE
    # idx_features_labels['RULE'] = 2*(idx_features_labels['RULE']-idx_features_labels['RULE'].min()).div(idx_features_labels['RULE'].max() - idx_features_labels['RULE'].min()) - 1

    # # Normalize AGE
    # idx_features_labels['AGE'] = 2*(idx_features_labels['AGE']-idx_features_labels['AGE'].min()).div(idx_features_labels['AGE'].max() - idx_features_labels['AGE'].min()) - 1

    # # Normalize TSERVD
    # idx_features_labels['TSERVD'] = 2*(idx_features_labels['TSERVD']-idx_features_labels['TSERVD'].min()).div(idx_features_labels['TSERVD'].max() - idx_features_labels['TSERVD'].min()) - 1

    # # Normalize FOLLOW
    # idx_features_labels['FOLLOW'] = 2*(idx_features_labels['FOLLOW']-idx_features_labels['FOLLOW'].min()).div(idx_features_labels['FOLLOW'].max() - idx_features_labels['FOLLOW'].min()) - 1

    # # Normalize TIME
    # idx_features_labels['TIME'] = 2*(idx_features_labels['TIME']-idx_features_labels['TIME'].min()).div(idx_features_labels['TIME'].max() - idx_features_labels['TIME'].min()) - 1

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens_idx = set(np.where(sens >= 0)[0])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(20)
    random.shuffle(idx_sens_train)
    # idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])
    torch.save([features, labels, idx_train, idx_val, idx_test, sens],'data/inf/bail_information.pt')
    # return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="data/ori/german/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

#    for i in range(idx_features_labels['PurposeOfLoan'].unique().shape[0]):
#        val = idx_features_labels['PurposeOfLoan'].unique()[i]
#        idx_features_labels['PurposeOfLoan'][idx_features_labels['PurposeOfLoan'] == val] = i

#    # Normalize LoanAmount
#    idx_features_labels['LoanAmount'] = 2*(idx_features_labels['LoanAmount']-idx_features_labels['LoanAmount'].min()).div(idx_features_labels['LoanAmount'].max() - idx_features_labels['LoanAmount'].min()) - 1
#
#    # Normalize Age
#    idx_features_labels['Age'] = 2*(idx_features_labels['Age']-idx_features_labels['Age'].min()).div(idx_features_labels['Age'].max() - idx_features_labels['Age'].min()) - 1
#
#    # Normalize LoanDuration
#    idx_features_labels['LoanDuration'] = 2*(idx_features_labels['LoanDuration']-idx_features_labels['LoanDuration'].min()).div(idx_features_labels['LoanDuration'].max() - idx_features_labels['LoanDuration'].min()) - 1
#
    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(20)
    random.shuffle(idx_sens_train)
    # idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    torch.save([features, labels, idx_train, idx_val, idx_test, sens],'data/inf/german_information.pt')
    # return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train

def build_relationship(feature, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(feature.T.T, feature.T.T)),
                              columns=feature.T.columns, index=feature.T.columns) 
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2] 
        neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig]) 
    idx_map =  np.array(idx_map) 
    return idx_map

def feature_normalize(feature): 
    '''sum_norm'''
    feature = np.array(feature)
    rowsum = feature.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return feature / rowsum

def train_val_test_split(labels,train_ratio=0.5,val_ratio=0.25,seed=20,label_number=1000):
    import random
    random.seed(seed)
    label_idx_0 = np.where(labels==0)[0]  #
    label_idx_1 = np.where(labels==1)[0]  # 
    random.shuffle(label_idx_0) 
    random.shuffle(label_idx_1)
    position1 = train_ratio
    position2 = train_ratio + val_ratio
    idx_train = np.append(label_idx_0[:min(int(position1 * len(label_idx_0)), label_number//2)], 
                          label_idx_1[:min(int(position1 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(position1 * len(label_idx_0)):int(position2 * len(label_idx_0))], 
                        label_idx_1[int(position1 * len(label_idx_1)):int(position2 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(position2 * len(label_idx_0)):],
                         label_idx_1[int(position2 * len(label_idx_1)):])
    print('train,val,test:',len(idx_train),len(idx_val),len(idx_test))
    return idx_train, idx_val, idx_test