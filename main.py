import yaml
import argparse
import torch
import torchmetrics
from sklearn.metrics import f1_score, roc_auc_score
from models import FUGNN
from utils import init_params, seed_everything
import time
import numpy as np
import pandas as pd
import torch.nn.functional as F
import warnings
import os
warnings.filterwarnings("ignore")


'''
model, dataset, Acc, SP, EO, metric_choose, seed, K, Max_cached, Max_allo, hidden_dim, lr, best epoch, nlayer, norm, train_num, val_num, test_num, 
val: acc, acc-sp-eo, loss 
'''

def main_worker(args, config, j):
    device=torch.device("cuda:"+str(args.cuda) if torch.cuda.is_available() else "cpu")
    max_acc1= None
    print(args, config)
    start = time.time()
    epoch = config['epoch']
    lr = config['lr']
    weight_decay = config['weight_decay']
    nclass = config['nclass']
    nlayer = config['nlayer']
    hidden_dim = config['hidden_dim']
    num_heads = config['num_heads']
    tran_dropout = config['tran_dropout']
    feat_dropout = config['feat_dropout']
    prop_dropout = config['prop_dropout']
    norm = config['norm']

    eignvalue, eignvector, feature = torch.load('data/eig/'+config['dataset']+'_'+str(j)+'_'+str(config['sens_idex'])+'.pt')
    # eignvalue, eignvector, feature = eignvalue, eignvector, feature
    _, labels, idx_train, idx_val, idx_test, sens = torch.load('data/inf/'+config['dataset']+'_information.pt')

    nfeat = feature.size(1)
    net = FUGNN(nclass, nfeat, nlayer, hidden_dim, num_heads, tran_dropout, feat_dropout, prop_dropout, norm)
    # print(net)
    net.apply(init_params)
    
    labels, idx_train, idx_val, idx_test, sens = labels.to(device), idx_train.to(device), idx_val.to(device), idx_test.to(device), sens.to(device)
    print('data len:',labels.shape,'train:',len(idx_train),'val:',len(idx_val),'test',len(idx_test))
    
    eignvalue, eignvector, feature = eignvalue.to(device), eignvector.to(device), feature.to(device)
    net=net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    res = []
    # min_loss = 100.0
    best_metric = -999998.0
    max_acc1=None
    new_metric = -999999.0
    
    best_metric_test = -999998.0
    max_acc1_test=None
    new_metric_test = -999999.0
    
    if args.metric==1: # acc
        print('metric: acc')
    elif args.metric==2: # loss
        print('metric: loss')
    elif args.metric==3: # -sp-eo
        print('metric: -sp-eo')
    elif args.metric==4: # val_acc-val_parity-val_equality
        print('metric: acc-sp-eo')

    
    counter = 0
    evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)
    end = time.time()
    print('success load data, time is:{:.3f}'.format(end-start))
    train_start = time.time()
    for idx in range(epoch):
        net.train()
        optimizer.zero_grad()
        logits = net(eignvalue, eignvector, feature)

        loss = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        net.eval()
        logits = net(eignvalue, eignvector, feature)

        val_loss = F.cross_entropy(logits[idx_val], labels[idx_val]).item()
        val_parity, val_equality = fair_metric(labels, sens, torch.argmax(logits, dim=1), idx_val)
        val_acc = evaluation(logits[idx_val].cpu(), labels[idx_val].cpu()).item()
        val_auc_roc = roc_auc_score(labels[idx_val].cpu().numpy(), F.softmax(logits,dim=1)[idx_val,1].detach().cpu().numpy())
        val_f1 = f1_score(labels[idx_val].cpu().numpy(),logits[idx_val].detach().cpu().argmax(dim=1))
        val_parity, val_equality = fair_metric(labels, sens, torch.argmax(logits, dim=1), idx_val)

        
        test_loss = F.cross_entropy(logits[idx_test], labels[idx_test]).item()   
        test_acc = evaluation(logits[idx_test].cpu(), labels[idx_test].cpu()).item()
        test_auc_roc = roc_auc_score(labels[idx_test].cpu().numpy(), F.softmax(logits,dim=1)[idx_test,1].detach().cpu().numpy())
        test_f1 = f1_score(labels[idx_test].cpu().numpy(),logits[idx_test].detach().cpu().argmax(dim=1))
        test_parity, test_equality = fair_metric(labels, sens, torch.argmax(logits, dim=1), idx_test)
        
        
        # acc, sp, eo, f1, auc, epoch
        res.append([100 * test_acc, 100 * test_parity, 100 * test_equality, test_f1, test_auc_roc,(idx+1)])

        # new_metric = (val_acc-val_parity-val_equality)
        if args.metric==1: # acc
            new_metric = val_acc
            new_metric_test = test_acc
        elif args.metric==2: # loss
            new_metric = -val_loss
            new_metric_test = -test_loss
        elif args.metric==3 and idx>100: # -sp-eo
            new_metric = (-val_parity-val_equality)
            new_metric_test = (-test_parity-test_equality)
        elif args.metric==4: # val_acc-val_parity-val_equality
            new_metric = (val_acc-val_parity-val_equality)
            new_metric_test = (test_acc-test_parity-test_equality)
            
        if new_metric > best_metric and (idx+1)>=10:
            best_metric = new_metric
            max_acc1 = res[-1]
            counter = 0 
        else:
            counter += 1
            
        if new_metric_test > best_metric_test and (idx+1)>=10:
            best_metric_test = new_metric_test
            max_acc1_test = res[-1]

        if (idx+1)%10==0:
            print('epoch:{:05d}, val_loss{:.4f}, test_acc:{:.4f}, aucroc:{:.4f}, f1:{:.4f}, parity:{:.4f}, equality:{:.4f}'.format(idx+1,val_loss,100 * test_acc,test_auc_roc,test_f1,100 * test_parity,100 * test_equality))
        
        if counter>200 and idx>500:
            train_end = time.time()
            train_time = (train_end-train_start)
            print('success train data, time is:{:.3f}'.format(train_time))
            break
    
    print('final_test_acc:', max_acc1[0], 'parity:',max_acc1[1],'equality:', max_acc1[2],'epoch:',idx+1)
    print('best_best_acc:', max_acc1_test[0], 'parity:',max_acc1_test[1],'equality:', max_acc1_test[2],'epoch:',idx+1)
    # model, dataset, Acc, SP, EO, metric_choose, seed, K, Max_cached, Max_allo, hidden_dim, lr, best epoch, nlayer, norm, train_num, val_num, test_num, 
    
    max_memory_cached = torch.cuda.max_memory_cached(device=device) / 1024 ** 2 
    max_memory_allocated = torch.cuda.max_memory_allocated(device=device) / 1024 ** 2
    print("Max memory cached:", max_memory_cached, "MB")
    print("Max memory allocated:", max_memory_allocated, "MB")
    # import os
    
    
def fair_metric(y, sens, output, idx):
    val_y = y[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

    pred_y = (output[idx].squeeze() > 0).type_as(y).cpu().numpy()

    parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1))

    return parity, equality

if __name__ == '__main__':
    start = time.time()
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--seed', type=int, default=0) 
    parser.add_argument('--cuda', type=int, default=0) 
    
    parser.add_argument('--dataset', default='credit')
    parser.add_argument('--image', type=int, default=0)
    parser.add_argument('--metric', type=int, default=4)
    parser.add_argument('--K', type=int, default=100) # 
    # args.metric=3
    # 1:acc
    # 2:-val_loss
    # 3:-sp-eo
    # 4:acc-sp-eo
    args = parser.parse_args()
    config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)[args.dataset]
    
    seed_everything(args.seed)
    print(config['dataset'])
    # args.K=config['k_start']
    
    # j=config['k_start']
    j = args.K
    main_worker(args, config, j)
    print('finish seed:', args.seed)
    
    
    end = time.time()
    total_time = end - start
    
    print('total time cost is {}s'.format(total_time))
    # log_save
    print('train over')
    # nohup python main_node_metric.py > result_pokec_z_seed0.log 2>&1 &
    # nohup python main_node_metric.py --seed 0 --dataset nba --K 1 --metric 4 > result_nba_k1_seed0.log 2>&1 &