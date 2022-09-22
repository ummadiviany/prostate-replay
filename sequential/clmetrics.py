import numpy as np

def print_cl_metrics(train_order, test_order, tr):

    tr = np.array(tr)
    new_tr = np.zeros(tr.shape)
    reorder_idxs = [train_order.index(x) for x in test_order]
    new_tr = tr[:, reorder_idxs]

    print(f"Average accuracy (ACC): {np.mean(new_tr[-1]):.2f}")
    N = len(train_order)
    a_arr = np.tril(new_tr, k=0)
    print(f"Average accuracy (A): {np.sum(a_arr/(N*(N+1)/2)):.2f}")
    
    fwt_arr = np.triu(new_tr, k=1)
    fwt = np.sum(fwt_arr)/((N-1)*N/2)
    print(f"Forward transfer (FWT): {fwt:.2f}")
    
    bwt_arr = [(new_tr[N-1][i] - new_tr[i][i]) for i in range(N-1)]
    bwt = (1/(N-1)) * np.sum(bwt_arr)
    print(f"Backward transfer (BWT): {bwt:.2f}")
    print(f"Forgetting: {-bwt:.2f}")


# metrics_dict = {}
# for i, m in enumerate(train_order):
#     metrics_dict[m] = {}
#     for j, n in enumerate(test_order):
#         metrics_dict[m][n] = tr[i][j]

if __name__ == "__main__":
    print("-"*50)
    # print("Metrics")
    
    # acc_arr = [metrics_dict[train_order[-1]][te] for te in metrics_dict[train_order[-1]]]
    # acc = np.mean(acc_arr)
    # print(f"Average accuracy: {acc:.2f}")
    
    
    # bwt_arr = [(metrics_dict[train_order[-1]][te] - metrics_dict[te][te]) for te in train_order[:-1]]
    # T = len(train_order)
    # bwt = (1/(T-1)) * np.sum(bwt_arr)
    # print(f"Backward transfer: {bwt:.2f}")
    # print(f"Forgetting: {-bwt:.2f}")
    
    train_order = ['decathlon', 'promise12', 'isbi', 'prostate158']
    test_order = ['prostate158',  'isbi', 'promise12', 'decathlon', ]

    seq_tr = [
        [36.7, 62.4, 58.7, 78.4],
        [65.2, 79.4, 88.9, 86.1],
        [44.3, 76.9, 45.3, 82.6],
        [83.1, 69.7, 31.9, 73.3],
        ]
    
    print_cl_metrics(train_order, test_order, seq_tr)
    print("-"*50)
    
    replay_tr = [
        [36.7, 62.4, 58.7, 78.4],
        [62.6, 77.6, 88.9, 86.1],
        [50.4, 77.2, 79.7, 82.1],
        [84.6, 82.3, 88.2, 84.2],
        ]
    
    print_cl_metrics(train_order, test_order, replay_tr)
    