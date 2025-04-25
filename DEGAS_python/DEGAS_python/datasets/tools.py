import pandas as pd
import numpy as np
import os
import math

# ##################### These functions are from original version of DEGAS #################################
# def resample(prc_cut,Y,train):
#     add = list()
#     rem = list()
#     train = np.squeeze(train)
#     colsums = np.sum(Y[train,:],axis=0);
#     cutoff = math.ceil(np.percentile(colsums,prc_cut));
#     for i in range(len(colsums)):
#         if colsums[i] == 0:
#             pass
#         elif colsums[i] < cutoff:
#             idx = np.squeeze(np.where(Y[train,i]>=1));
#             choice = np.random.choice(train[idx],int(cutoff-colsums[i]))
#             add = add + choice.tolist();
#         elif colsums[i] == cutoff:
#             pass
#         else:
#             idx = np.squeeze(np.where(Y[train,i]>=1));
#             choice = np.random.choice(train[idx],int(colsums[i]-cutoff),replace=False)
#             rem = rem + choice.tolist()
#     return list(set(train)-set(rem))+add

# def resample_mixGamma(X, Y, train, nsamp, depth):
#     # from inst/tools of DEGAS
#     add = list()
#     train = np.squeeze(train)
#     colsums = np.sum(Y[train,:],axis=0)
#     samp_per_class = round(nsamp/len(colsums))
#     idx = list()
#     for i in range(len(colsums)):
#         idx = idx + [np.squeeze(np.where(Y[train,i]>=1)).tolist()];
#         if samp_per_class > colsums[i]:
#             choice = np.random.choice(train[idx[i]],int(samp_per_class),replace=True)
#         else:
#             choice = np.random.choice(train[idx[i]],int(samp_per_class),replace=False)
#         add = add + choice.tolist()
#     tmpX = np.zeros([nsamp,X.shape[1]])
#     tmpY = np.zeros([nsamp,Y.shape[1]])
#     for i in range(nsamp):
#         percBinom = np.random.gamma(shape=1,size=len(colsums))
#         percBinom = percBinom/sum(percBinom)
#         intBinom = np.round(percBinom*depth)
#         tmpIdx = list()
#         for j in range(len(colsums)):
#             if int(intBinom[j]) > colsums[j]:
#                 tmpIdx = tmpIdx + np.random.choice(train[idx[j]], int(intBinom[j]),replace=True).tolist()
#             else:
#                 tmpIdx = tmpIdx + np.random.choice(train[idx[j]], int(intBinom[j]),replace=False).tolist()
#         tmpX[i,:] = np.mean(X[tmpIdx,:],axis=0)+1e-3
#         tmpY[i,:] = intBinom/sum(intBinom)
#     #scaler = preprocessing.MinMaxScaler()        # CHANGED 20201213
#     #tmpX = np.transpose(scaler.fit_transform(np.transpose(zscore(tmpX,axis=0))))     # CHANGED 20201212
#     #return(tmpX,tmpY)		#CHANGED 20201211
#     return(np.concatenate((X[add,:],tmpX), axis=0),np.concatenate((Y[add,:],tmpY)))		#CHANGED 20201211


# Ziyu's sampling method
    
def balance_sampling(labels, batch_size, random_seed):
    np.random.seed(random_seed)
    assert len(labels.shape) == 1
    label_df = pd.DataFrame({
            "labels": list(labels),
            "indices": list(range(len(labels)))
        })
    categories = np.unique(label_df["labels"])
    balance_samples = []
    for category in categories:
        sub_label_df = label_df[label_df["labels"] == category]
        balance_samples += list(np.random.choice(sub_label_df["indices"].values, batch_size // len(categories), replace = True))
    return balance_samples
