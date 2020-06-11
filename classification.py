import evaluation
import data_process
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from sklearn import model_selection, linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
import lightgbm as lgb


def construct_model(dim, L1, L2):
    model = Sequential()
    model.add(Dense(units=L1, activation='tanh', input_dim=dim, kernel_regularizer=l2(0.02),kernel_initializer='random_uniform'))
    if L2 != 0:
        model.add(Dense(units=L2, activation='tanh', kernel_regularizer=l2(0.02),kernel_initializer='random_uniform'))
    model.add(Dense(1, activation='sigmoid',kernel_initializer='random_uniform'))
    learning_rate = 0.01
    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
    return model


def train_nn(X,Y,L1,L2,k=5):
    k_fold = model_selection.KFold(n_splits=k)
    callbacks = [EarlyStopping(monitor='loss', patience=3),
                 ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_best_only=True)]
    acc=0
    for train_idx, test_idx in k_fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        train_val_split = int(0.8 * len(X_train))
        X_train, X_val = X_train[:train_val_split, :], X_train[train_val_split:, :]
        Y_train, Y_val = Y_train[:train_val_split], Y_train[train_val_split:]
        model = construct_model(X.shape[1], L1, L2)
        model.fit(X_train,
                  Y_train,
                  verbose=0,
                  epochs=500,
                  callbacks=callbacks,
                  validation_data=(X_val, Y_val),
                  batch_size=100)
        Y_predict = model.predict_classes(X_test)
        cur_acc=evaluation.get_accuracy(Y_predict,Y_test)
        print(str(cur_acc))
        acc+=cur_acc
    acc /= k
    print('nn: '+str(acc))


def simple_nn(X_train,Y_train,X_test,L1=10,L2=10,type='hard'):
    callbacks = [EarlyStopping(monitor='loss', patience=3),
                 ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_best_only=True)]
    model = construct_model(X_train.shape[1], L1, L2)
    model.fit(X_train,
              Y_train,
              verbose=0,
              epochs=500,
              callbacks=callbacks,
              validation_split=0.2,
              batch_size=100)
    if type is 'hard':
        Y_predict = model.predict_classes(X_test)
    else:
        Y_predict = model.predict(X_test)
    return Y_predict


# # lightgbm 的predict 返回的不是class, 所以需要重新定义evaluate.
# def evaluate_lightgbm(predict, Y):
#     N=len(Y)
#     correct = 0
#     for i in range(N):
#         if predict[i] >= 0.5:
#             if Y[i]==1:
#                 correct+=1
#         else:
#             if Y[i]==0:
#                 correct +=1
#     return correct/N


# logistic
def train_logistic(X, Y,k=10):
    k_fold = model_selection.KFold(n_splits=k, shuffle=True)
    acc_all,auc_all,f1_all = 0,0,0
    for train_idx, test_idx in k_fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        clf = linear_model.LogisticRegression(penalty='none',max_iter=1000,solver='saga').fit(X, Y.ravel())
        predict = clf.predict(X_test)
        acc,auc,f1 =evaluation.eval_with_plot(predict,Y_test)
        acc_all += acc
        auc_all +=auc
        f1_all+=f1
    acc_all/=k
    auc_all/=k
    f1_all/=k
    print("acc: " + str(acc_all) + ' auc: ' + str(auc_all) + 'f1 ' + str(f1_all))


def simple_LR(X_train,Y_train,X_test):
    clf = linear_model.LogisticRegression(max_iter=1000).fit(X_train, Y_train.ravel())
    predict = clf.predict(X_test)
    return predict


# svm, n is chose to be 5.
def train_svc(X, Y,k=10):
    k_fold = model_selection.KFold(n_splits=k, shuffle=True)
    acc_all, auc_all, f1_all = 0, 0, 0
    for train_idx, test_idx in k_fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        clf=svm.SVC()
        clf.fit(X_train,Y_train.ravel())
        predict = clf.predict(X_test)
        acc, auc, f1 = evaluation.eval(predict, Y_test)
        acc_all += acc
        auc_all += auc
        f1_all += f1
    acc_all /= k
    auc_all /= k
    f1_all /= k
    print("acc: " + str(acc_all) + ' auc: ' + str(auc_all) + 'f1 ' + str(f1_all))


def simple_svm(X_train,Y_train,X_test):
    clf = svm.SVC()
    clf.fit(X_train, Y_train.ravel())
    predict =  clf.predict(X_test)
    return predict


#lightgbm 0.998
def train_lightgbm(X, Y,k=10):
    k_fold = model_selection.KFold(n_splits=k, shuffle=True)
    acc_all, auc_all, f1_all = 0, 0, 0
    for train_idx, test_idx in k_fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        lgb_train = lgb.Dataset(X_train, Y_train, free_raw_data=False)
        lgb_test = lgb.Dataset(X_test, Y_test, reference=lgb_train, free_raw_data=False)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss', 'auc', 'acc'},
            'num_leaves': 5,
            'max_depth': 6,
            'min_data_in_leaf': 450,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.95,
            'bagging_freq': 5,
            'lambda_l1': 1,
            'lambda_l2': 0.001,
            'min_gain_to_split': 0.2,
            'verbose': 5,
            'is_unbalance': True
        }
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=lgb_test,
                        early_stopping_rounds=500)
        predict = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        predict = np.round(predict)
        acc, auc, f1 = evaluation.eval(predict, Y_test)
        acc_all += acc
        auc_all += auc
        f1_all += f1
    acc_all /= k
    auc_all /= k
    f1_all /= k
    print("acc: " + str(acc_all) + ' auc: ' + str(auc_all) + 'f1 ' + str(f1_all))


def simple_lightgbm(X_train, Y_train,X_test,  Y_test):
    lgb_train = lgb.Dataset(X_train, Y_train, free_raw_data=False)
    lgb_test = lgb.Dataset(X_test, Y_test, reference=lgb_train, free_raw_data=False)
    params ={
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss', 'auc', 'acc'},
            'num_leaves': 5,
            'max_depth': 6,
            'min_data_in_leaf': 450,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.95,
            'bagging_freq': 5,
            'lambda_l1': 1,
            'lambda_l2': 0.001,
            'min_gain_to_split': 0.2,
            'verbose': 5,
            'is_unbalance': True
        }
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=lgb_test,
                    early_stopping_rounds=500)
    predict = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    predict = np.round(predict)
    return predict


# Adaboost 0.94
def train_adaboost(X,Y,k=10):
    k_fold = model_selection.KFold(n_splits=k, shuffle=True)
    acc_all, auc_all, f1_all = 0, 0, 0
    clf = AdaBoostClassifier()
    for train_idx, test_idx in k_fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        clf.fit(X_train,Y_train)
        predict = clf.predict(X_test)
        acc, auc, f1 = evaluation.eval(predict, Y_test)
        acc_all += acc
        auc_all += auc
        f1_all += f1
    acc_all /= k
    auc_all /= k
    f1_all /= k
    print("acc: " + str(acc_all) + ' auc: ' + str(auc_all) + 'f1 ' + str(f1_all))


def simple_adaboost(X_train,Y_train,X_test):
    clf = AdaBoostClassifier()
    clf.fit(X_train, Y_train)
    predict = clf.predict(X_test)
    return predict


# random forest 0.85
def train_randomForest(X,Y,k=10):
    k_fold = model_selection.KFold(n_splits=k, shuffle=True)
    acc_all, auc_all, f1_all = 0, 0, 0
    clf = RandomForestClassifier(max_depth=6, max_leaf_nodes=5)
    for train_idx, test_idx in k_fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        clf.fit(X_train,Y_train)
        predict = clf.predict(X_test)
        acc, auc, f1 = evaluation.eval_with_plot(predict, Y_test)
        acc_all += acc
        auc_all += auc
        f1_all += f1
        print("acc: " + str(acc) + ' auc: ' + str(auc) + 'f1 ' + str(f1))
    acc_all /= k
    auc_all /= k
    f1_all /= k
    print("acc: " + str(acc_all) + ' auc: ' + str(auc_all) + 'f1 ' + str(f1_all))


def simple_randomForest(X_train,Y_train,X_test):
    clf = RandomForestClassifier(max_depth=6, max_leaf_nodes=5)
    clf.fit(X_train, Y_train)
    Y_p = clf.predict(X_test)
    return Y_p


def vote_ensemble_result(output):
    N ,y_num= output.shape[0],output.shape[1]
    target = N/2
    accum = np.sum(output,axis=0)
    res=[]
    for i in range(y_num):
        if accum[i] >=target:
            res.append(1)
        else:
            res.append(0)
    return np.array(res)


def vote_ensemble_model(X_train,Y_train,X_test,Y_test):
    N = X_test.shape[0]
    # nn
    Y_nn = simple_nn(X_train,Y_train,X_test).reshape(1,N)
    # svm
    Y_lr = simple_LR(X_train,Y_train,X_test).reshape(1,N)
    # lr
    Y_ada = simple_adaboost(X_train,Y_train,X_test).reshape(1,N)
    # random forest
    Y_rf = simple_randomForest(X_train,Y_train,X_test).reshape(1,N)
    # adaboost
    Y_svm = simple_svm(X_train,Y_train,X_test).reshape(1,N)
    # lightGBM
    Y_light = simple_lightgbm(X_train,Y_train,X_test,Y_test).reshape(1,N)
    Y_all = np.concatenate([Y_nn,Y_lr,Y_ada,Y_rf,Y_svm,Y_light],axis=0)
    return vote_ensemble_result(Y_all)


def k_fold_ensemble(X,Y,k=5):
    k_fold = model_selection.KFold(n_splits=k, shuffle=True)
    acc_all = 0
    for train_idx,test_idx in k_fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        y_predict = vote_ensemble_model(X_train,Y_train,X_test)
        cur_acc = evaluation.get_accuracy(y_predict,Y_test)
        print(str(cur_acc))
        acc_all+=cur_acc
    acc_all/=k
    print('acc '+str(acc_all))


if __name__ == '__main__':
    data, result, Y_bin = data_process.get_data_from_csv()
    N = len(Y_bin)
    X = data
    X = data_process.retain_pca(X, 0.9999)
    Y = Y_bin.reshape(N)  # for lightgbm, the label is required to be 1D array.
    split = int(0.8*N)
    X_train,Y_train=X[:split,:],Y[:split]
    X_test,Y_test = X[split:,:],Y[split:]
    # predict = simple_nn(X_train,Y_train,X_test)
    # evaluation.eval_with_plot(predict,Y_test)
    Y_predict = vote_ensemble_model(X_train,Y_train,X_test,Y_test)
    evaluation.eval_with_plot(Y_predict,Y_test)
    # train_logistic(X, Y_bin, 10)
    # simple_svm(X_train,Y_train, X_test,Y_test)
    # train_lightgbm(X, Y)
    # train_adaboost(X, Y)
    # train_randomForest(X, Y)
    # train_·nn(X,Y,10,10)


