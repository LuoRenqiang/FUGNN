import os
import scipy as sp
import numpy as np
import pandas as pd
import torch
from numpy import int64
import yaml
from utils import load_dataset,load_bail,load_german
from scipy.spatial import distance_matrix
import scipy.sparse as sp
# import cupy
# from cupyx.scipy.sparse.linalg._eigen import eigsh
import time

def feature_normalize(feature):
    feature = np.array(feature)
    rowsum = feature.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return feature / rowsum

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns) #distance_matrix ，ijij
    df_euclid = df_euclid.to_numpy() #
    idx_map = [] #list
    for ind in range(df_euclid.shape[0]): #numpy.shape，numpy.shape[0]
        max_sim = np.sort(df_euclid[ind, :])[-2] #
        neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0] #tresh*max_sim
        import random
        random.seed(912)
        random.shuffle(neig_id) #neig
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig]) #neigidx_map
    # print('building edge relationship complete')
    idx_map =  np.array(idx_map)
    
    return idx_map



def generate_node_data(dataset, k_start, k_end, k_jump, sens_idex,self_loop=False):
    print('k_start:',k_start,'k_end:',k_end,'k_jump:',k_jump)
    start = time.time()
    if dataset in ['nba']:# no leak
        # edge_df = pd.read_csv('data/ori/nba/' + 'nba_relationship.txt', sep='\t')
        edges_unordered = np.genfromtxt('data/ori/nba/' + 'nba_relationship.txt').astype('int')
        node_df = pd.read_csv(os.path.join('data/ori/nba/', 'nba.csv'))
        
        print('load edge data')
        y = node_df["SALARY"].values
        labels = y
        adj_start = time.time()
        feature = node_df[node_df.columns[2:]]

        if sens_idex:
            feature = feature.drop(columns = ["country"])
        
        idx = node_df['user_id'].values # for relations
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #edges_unordered
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #sp.coo_matrix（csr_matrix）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #
        if self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye1
        else:
            print('no add self-loop')
        adj_end = time.time()
        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        # print('adj created!')
        feature = np.array(feature)
        feature = feature_normalize(feature)
        for i in range(k_start, k_end, k_jump):
            
            # eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k = i)
            # eignvalue, eignvector = eigsh(adj, which='LM', k = i)
            eigsh_start = time.time()
            # eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k=i)
            # eignvalue, eignvector = np.eigh(adj)
            eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k=i)
            eigsh_end = time.time()
            print('eigsh time is {:.3f}'.format((eigsh_end-eigsh_start)))
            eignvalue = torch.FloatTensor(eignvalue)
            eignvector = torch.FloatTensor(eignvector)
            feature = torch.FloatTensor(feature)
            # print(eignvalue)
            torch.save([eignvalue, eignvector, feature],  'data/eig/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')
            # torch.save([eignvalue, eignvector, feature],  'data/eig_test/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')
    
    elif dataset in ['region_job']: #no
        # edge_df = pd.read_csv('data/ori/pokec/' + 'region_job_relationship.txt', sep='\t')
        edges_unordered = np.genfromtxt('data/ori/pokec/' +'region_job_relationship.txt').astype('int')
        # node_df = pd.read_csv(os.path.join('data/ori/pokec/','region_job.csv'))
        print('load edge data')
        # y = node_df["I_am_working_in_field"].values
        # feature = node_df[node_df.columns[2:]]
        predict_attr = 'I_am_working_in_field'
        sens_attr = 'region'
        # ----
        print('Loading {} dataset'.format(dataset))

        idx_features_labels = pd.read_csv(os.path.join('data/ori/pokec/','region_job.csv'))
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        #header.remove(sens_attr)
        header.remove(predict_attr)


        # feature = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        # labels = idx_features_labels[predict_attr].values
        feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values #predict_attr
        #-----
        adj_start = time.time()
        if sens_idex:
            feature = feature.drop(columns = ["region"])

        # idx = node_df['user_id'].values # for relations
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #edges_unordered
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #sp.coo_matrix（csr_matrix）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #
        if self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye1
        else:
            print('no add self-loop')
        adj_end = time.time()
        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        # print('create adj')
        # feature = np.array(feature)
        # feature = feature_normalize(feature)
        for i in range(k_start, k_end, k_jump):
            # eignvalue, eignvector = sp.linalg.eign(adj, which='LM', k=i)
            eigsh_start = time.time()
            eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k=i)
            eigsh_end = time.time()
            print('create eignvalue and eignvector')
            print('eigsh time is {:.3f}'.format((eigsh_end-eigsh_start)))
            # eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k=i)
            # print('create eignvalue and eignvector')
            eignvalue = torch.FloatTensor(eignvalue)
            eignvector = torch.FloatTensor(eignvector)
            feature = torch.FloatTensor(feature)
            # print(eignvalue)

            torch.save([eignvalue, eignvector, feature],  'data/eig/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')
            # torch.save([eignvalue, eignvector, feature],  'data/eig_test/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')
    elif dataset in ['region_job_2']:
        
        edges_unordered = np.genfromtxt('data/ori/pokec/' +'region_job_2_relationship.txt').astype('int')

        # edge_df = pd.read_csv('data/ori/pokec/' + 'region_job_2_relationship.txt', sep='\t')
        # node_df = pd.read_csv(os.path.join('data/ori/pokec/','region_job_2.csv')) 
        idx_features_labels = pd.read_csv(os.path.join('data/ori/pokec/','region_job_2.csv'))
        predict_attr = 'I_am_working_in_field'
        sens_attr = 'region'
        print('load edge data')
        header = list(idx_features_labels.columns)
        header.remove("user_id")
        header.remove(predict_attr)
        
        feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values #predict_attr
        
        # y = idx_features_labels["I_am_working_in_field"].values
        # labels = y
        adj_start = time.time()
        # feature = node_df[node_df.columns[2:]]

        if sens_idex:
            feature = feature.drop(columns = ["region"])
        
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #edges_unordered
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #sp.coo_matrix（csr_matrix）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #
        if self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye1
        else:
            print('no add self-loop')
        adj_end = time.time()
        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        
        # feature = np.array(feature)
        # feature = feature_normalize(feature)
        for i in range(k_start, k_end, k_jump):
            eigsh_start = time.time()
            eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k=i)
            eigsh_end = time.time()
            print('create eignvalue and eignvector')
            print('eigsh time is {:.3f}'.format((eigsh_end-eigsh_start)))
            
            eignvalue = torch.FloatTensor(eignvalue)
            eignvector = torch.FloatTensor(eignvector)
            feature = torch.FloatTensor(feature)
            torch.save([eignvalue, eignvector, feature],  'data/eig/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')
            # torch.save([eignvalue, eignvector, feature],  'data/eig_test/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')

    elif dataset in ['credit']:
        # dataset='credit'
        path='./data/ori/credit/'
        sens_attr="Age"
        predict_attr="NoDefaultNextMonth"
        # label_number=1000
        print('Loading {} dataset from {}'.format(dataset, path))
        idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset))) #"{}.csv".format(“XXX”)"XXX.csv"，os.path.join(A,B)AB
        header = list(idx_features_labels.columns) #list
        header.remove(predict_attr) #header.remove
        header.remove('Single')
        # build relationship
        if os.path.exists(f'{path}/{dataset}_edges.txt'): #os.path.exists,f，
            edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int') #np.genfromtxttxt，astype
        else:
            edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7) #buildedge
            np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered) #
        feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values #predict_attr
        
        adj_start = time.time()
        idx = np.arange(feature.shape[0]) #0features：{0，1，2，...，features.shape[0]-1}
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #edges_unordered
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #sp.coo_matrix（csr_matrix）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #
        if self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye1
        else:
            print('no add self-loop')
        adj_end = time.time()
        
        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        for i in range(k_start, k_end, k_jump):
            eigsh_start = time.time()
            eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k=i)
            eigsh_end = time.time()
            print('create eignvalue and eignvector')
            print('eigsh time is {:.3f}'.format((eigsh_end-eigsh_start)))
            # eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k=i)
            # print('create eignvalue and eignvector')
            eignvalue = torch.FloatTensor(eignvalue)
            eignvector = torch.FloatTensor(eignvector)
            feature = torch.FloatTensor(feature)

            torch.save([eignvalue, eignvector, feature],  'data/eig/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')
            # torch.save([eignvalue, eignvector, feature],  'data/eig_test/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')

    elif dataset in ['income']:
        path='./data/ori/income/'
        sens_attr="race"
        predict_attr="income"
        print('Loading {} dataset from {}'.format(dataset, path))
        idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset))) #"{}.csv".format(“XXX”)"XXX.csv"，os.path.join(A,B)AB
        header = list(idx_features_labels.columns) #list
        header.remove(predict_attr) #header.remove
        # header.remove('Single')
        # header = list(idx_features_labels.columns)
        # header.remove(predict_attr)
        # build relationship
        if os.path.exists(f'{path}/{dataset}_edges.txt'): #os.path.exists,f，
            edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int') #np.genfromtxttxt，astype
        # else:
        #     edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7) #buildedge
            # np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered) #
        feature=feature_normalize(idx_features_labels[header])
        labels = idx_features_labels[predict_attr].values #predict_attr

        adj_start = time.time()
        idx = np.arange(feature.shape[0]) #0features：{0，1，2，...，features.shape[0]-1}
        idx_map = {j: i for i, j in enumerate(idx)} #{0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape) #edges_unordered
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32) #sp.coo_matrix（csr_matrix）
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #
        if self_loop:
            adj = adj + sp.eye(adj.shape[0]) #sp.eye1
        adj_end = time.time()

        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        for i in range(k_start, k_end, k_jump):
            eigsh_start = time.time()
            eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k=i)
            eigsh_end = time.time()
            print('create eignvalue and eignvector')
            print('eigsh time is {:.3f}'.format((eigsh_end-eigsh_start)))
            # eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k=i)
            # print('create eignvalue and eignvector')
            eignvalue = torch.FloatTensor(eignvalue)
            eignvector = torch.FloatTensor(eignvector)
            feature = torch.FloatTensor(feature)

            torch.save([eignvalue, eignvector, feature],  'data/eig/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')
            # torch.save([eignvalue, eignvector, feature],  'data/eig_test/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')
            
    elif dataset in ['bail']:
        path='./data/ori/bail/'
        sens_attr="WHITE"
        predict_attr="RECID"
        print('Loading {} dataset from {}'.format(dataset, path))
        idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        if os.path.exists(f'{path}/{dataset}_edges.txt'):
            edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
        else:
            edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
            np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        
        labels = idx_features_labels[predict_attr].values
        adj_start = time.time()
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
        if self_loop:
            adj = adj + sp.eye(adj.shape[0])
        adj_end = time.time()
        print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        feature = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
            
        # print('create adj time is {:.3f}'.format((adj_end-adj_start)))
        
        for i in range(k_start, k_end, k_jump):
            eigsh_start = time.time()
            eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k=i)
            eigsh_end = time.time()
            print('create eignvalue and eignvector')
            print('eigsh time is {:.3f}'.format((eigsh_end-eigsh_start)))
            # eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k=i)
            # print('create eignvalue and eignvector')
            eignvalue = torch.FloatTensor(eignvalue)
            eignvector = torch.FloatTensor(eignvector)
            feature = torch.FloatTensor(feature)

            torch.save([eignvalue, eignvector, feature],  'data/eig/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')
            
    elif dataset in ['german']:
        path='./data/ori/german/'
        sens_attr="Gender"
        predict_attr="GoodCustomer"
        idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove('OtherLoansAtStore')
        header.remove('PurposeOfLoan')
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0
        
        if os.path.exists(f'{path}/{dataset}_edges.txt'):
            edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
        else:
            edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
            np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        labels[labels == -1] = 0
        adj_start = time.time()
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
        if self_loop:
            adj = adj + sp.eye(adj.shape[0])

        feature = torch.FloatTensor(np.array(features.todense()))
        # labels = torch.LongTensor(labels)

        for i in range(k_start, k_end, k_jump):
            eigsh_start = time.time()
            eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k=i)
            eigsh_end = time.time()
            print('create eignvalue and eignvector')
            print('eigsh time is {:.3f}'.format((eigsh_end-eigsh_start)))
            # eignvalue, eignvector = sp.linalg.eigsh(adj, which='LM', k=i)
            # print('create eignvalue and eignvector')
            eignvalue = torch.FloatTensor(eignvalue)
            eignvector = torch.FloatTensor(eignvector)
            feature = torch.FloatTensor(feature)

            torch.save([eignvalue, eignvector, feature],  'data/eig/'+dataset+'_'+str(i)+'_'+str(sens_idex)+'.pt')
    end = time.time()
    print('generate_node_data time cost is:{:.3f}'.format((end-start)))

if __name__ == '__main__':
    # datasets=['pokec_z']
    # for i in datasets:
    #     config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)[i]
    #     # eig
    #     generate_node_data(config['dataset'], config['k_start'],config['k_end'], config['k_jump'], config['sens_idex'])
    #     # inf
    #     load_dataset(config['dataset'], config['sens_attr'], config['predict_attr'], config['path'],config['label_number'])
    # pokec_z pokec_n nba credit income bail german
    datasets=['credit']
    for i in datasets: # 
        # start = time.time()
        config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)[i]  
        # for eig
        generate_node_data(config['dataset'], 10, 11, 1, config['sens_idex'], self_loop=False)
        
        # for inf nba credit income pokec_z pokec_n
        load_dataset(config['dataset'], config['sens_attr'], config['predict_attr'], config['path'],config['label_number'],test_idx=False)
        # for inf german and bail 
        # load_german(i,"Gender","GoodCustomer","data/ori/german/",1000)
        # load_bail(i,"WHITE","RECID","data/ori/bail/",1000)
        
        # end = time.time()
        # total_time = end - start
        # print('total time cost is {:.3f}s'.format(total_time))
