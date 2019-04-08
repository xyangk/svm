#coding:utf-8

import time
import numpy as np


def selectJ(i, n):
    j = i
    while j == i:
        j = np.random.randint(0, n)
    return j

def kernel(x, y, k='linear'):
    if k == 'linear':
        return np.dot(x, y)

def simple_smo(data, label, C, epoch, tol=0.001):
    """
    data: n*m n样本数， m特征数
    label: n*1 标签， 正负1
    C: 软间隔问题中的对误分类的惩罚参数
    tol: 允许在一定误差范围内满足KKT条件
    epoch: 遍历数据集的次数
    """
    n, m = data.shape 
    alpha = np.zeros((n, 1))
    iter = 0
    b = 0
    while iter < epoch:
        num_alpha_changed = 0
        for i in range(n):
            gi = float(np.dot(np.multiply(alpha ,label).T, np.dot(data, data[i,:].T))) + b
            yi = label[i]
            Ei = gi - yi
            ai = alpha[i] #alpha_i
            # 判断该点是否违背KKT条件
            if (yi*Ei < -tol and ai < C) or (yi*Ei > tol and ai > 0):
                #随机选择第二个拉格朗日乘子，j != i
                j = selectJ(i, n)
                gj = float(np.dot(np.multiply(alpha ,label).T, np.dot(data, data[j,:]))) + b
                yj = label[j]
                Ej = gj - yj
                aj = alpha[j] # alpgha_j

                # 计算alpha的下限L 和 上限H
                if yi != yj:
                    L = max(0, aj-ai)
                    H = min(C, C + aj-ai)
                else:
                    L = max(0, aj + ai -C)
                    H = min(C, aj + ai)
                if L == H:
                    continue
                kii = np.dot(data[i,:], data[i,:].T) # K(x_i, x_i)
                kjj = np.dot(data[j,:], data[j,:].T)
                kij = np.dot(data[i,:], data[j,:].T)
                eta = kii + kjj - 2*kij
                if eta > 0:
                    aj_ = aj + yj * (Ei-Ej) / eta
                    if aj_ > H:
                        aj_ = H
                    elif aj_ < L:
                        aj_ = L
                    else:
                        pass
                else:
                    continue
                alpha[j] = aj_
                if abs(aj_ - aj) < 0.00001:
                    continue # 变化太小时不更新
                    # pass
                ai_ = ai + yi*yj *(aj_ - aj)
                alpha[i] = ai_

                num_alpha_changed += 1

                b_i = b - Ei - yi*kii*(ai_ - ai) - yj*kij*(aj_ - aj)
                b_j = b - Ej - yi*kij*(ai_ - ai) - yj*kjj*(aj_ - aj)
                if ai_ > 0 and ai_ < C:
                    b_new = b_i
                elif aj_ > 0 and aj_ < C:
                    b_new = b_j
                else:
                    b_new = (b_i + b_j) / 2
                
                b = b_new
                # print("ai: {}, aj: {}, b: {}".format(ai_, aj_, b_new))

        #当所有点都满足KKT条件时退出
        if num_alpha_changed == 0:
            iter += 1
        else:
            iter = 0
            # iter += 1
    return alpha, b

def train(data, label, C, epoch, tol=0.001):
    alpha, b = simple_smo(data, label, C, epoch, tol=tol)
    w = np.dot(np.multiply(alpha, label).T, data) #1 * m

    pred_x = np.sign(np.dot(x, w.T) + b)
    acc = np.sum(pred_x == label) / label.shape[0]
    print('acc:',acc)
    return w, b

    

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC

    x, y = make_blobs(n_samples=100, n_features=2, centers=2)#, random_state=1)
    y = np.reshape(y, (-1, 1))
    y[y<1] = -1
    # print(y)
    st = time.time()
    w, b = train(x, y, 0.6, 40)
    print(w, b)
    et = time.time()
    print('cost time: {}'.format(et - st))

    #sklearn svm
    clf = SVC(gamma='auto', C=0.6, kernel='linear', max_iter=40)
    clf.fit(x, np.squeeze(y))
    print(clf.score(x, np.squeeze(y)))
    w_sk = np.squeeze(clf.coef_ )
    b_sk = clf.intercept_ 
    print('sklearn param:, ', w_sk, b_sk)

    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y))
    
    # 直线方程 w_1 * x_1 + w_2 * x_2 + b = 0, 那么 x_2 = -(w_1*x_1+b)/w_2
    w = np.squeeze(w)
    x_1s = np.array([np.min(x[:, 0]), np.max(x[:, 0])])
    x_2s = -(x_1s * w[0] + b) / w[1]
    plt.plot(x_1s, x_2s, color='b')

    x_2sk = -(x_1s * w_sk[0] + b_sk) / w_sk[1]
    plt.plot(x_1s, x_2sk, color='r')

    plt.show()
