#coding:utf-8

import time
import numpy as np

class SVM(object):
    # def __init__(self, C, epoch, tol=0.001):
    #     self.alpha = np.zeros((n, 1))

    def selectJ(self, i):
        # j = i
        # while j == i:
        #     j = np.random.randint(0, n)
        # return j
        errs = np.abs(self.E_all - self.E_all[i])
        return np.argmax(errs)

    def kernel(self, x, y, k='linear'):
        if k == 'linear':
            return np.dot(x, y)

    def simple_smo(self, data, label, C, epoch, tol=0.001):
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
        data_matrix = np.dot(data, data.T) # n*n
        self.E_all = np.dot(np.multiply(alpha ,label).T, np.dot(data, data.T)).T + b - label #1*n x n*n = 1*n
        while iter < epoch:
            num_alpha_changed = 0
            for i in range(n):
                # gi = float(np.dot(np.multiply(alpha ,label).T, data_matrix[:, i])) + b
                yi = label[i]
                # Ei = gi - yi
                Ei = self.E_all[i]
                ai = float(alpha[i]) #alpha_i
                # 判断该点是否违背KKT条件
                if (yi*Ei < -tol and ai < C) or (yi*Ei > tol and ai > 0):
                    #随机选择第二个拉格朗日乘子，j != i
                    j = self.selectJ(i)
                    # gj = float(np.dot(np.multiply(alpha ,label).T, data_matrix[:, j])) + b
                    yj = label[j]
                    # Ej = gj - yj
                    Ej = self.E_all[j]
                    aj = float(alpha[j]) # alpgha_j

                    # 计算alpha的下限L 和 上限H
                    if yi != yj:
                        L = max(0, aj-ai)
                        H = min(C, C + aj-ai)
                    else:
                        L = max(0, aj + ai -C)
                        H = min(C, aj + ai)
                    if L == H:
                        # print('L==H')
                        continue
                    kii = data_matrix[i, i] # K(x_i, x_i)
                    kjj = data_matrix[j, j]
                    kij = data_matrix[i, j]
                    eta = kii + kjj - 2*kij
                    if eta > 0:
                        kkk = yj * (Ei-Ej) / eta
                        aj_ = aj + yj * (Ei-Ej) / eta
                        if aj_ > H:
                            aj_ = H
                        elif aj_ < L:
                            aj_ = L
                        else:
                            pass
                    else:
                        print('eta<0')
                        continue
                    alpha[j] = aj_
                    # print(aj_ -aj)
                    if abs(aj_ - aj) < 0.00001:
                        # print('abs(aj_ - aj) < 0.00001')
                        continue # 变化太小时不更新
                        # pass
                    # print('update J')
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

                    # update error matrix
                    self.E_all = np.dot(np.multiply(alpha ,label).T, np.dot(data, data.T)).T + b - label

            #当所有点都满足KKT条件时退出
            print('num_alpha_changed', num_alpha_changed)
            if num_alpha_changed == 0:
                iter += 1
            else:
                iter = 0
                # iter += 1
        return alpha, b

    def train(self, data, label, C, epoch, tol=0.001):
        alpha, b = self.simple_smo(data, label, C, epoch, tol=tol)
        w = np.dot(np.multiply(alpha, label).T, data) #1 * m

        pred_x = np.sign(np.dot(x, w.T) + b)
        acc = np.sum(pred_x == label) / label.shape[0]
        print('acc:',acc)
        return w, b

        

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC

    x, y = make_blobs(n_samples=1000, n_features=2, centers=2)#, random_state=1)
    y = np.reshape(y, (-1, 1))
    y[y<1] = -1
    # print(y)
    st = time.time()
    svm = SVM()
    w, b = svm.train(x, y, 0.6, 40)
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
    plt.plot(x_1s, x_2s, color='b') #my svm

    x_2sk = -(x_1s * w_sk[0] + b_sk) / w_sk[1]
    plt.plot(x_1s, x_2sk, color='r') #sklearn

    plt.show()
