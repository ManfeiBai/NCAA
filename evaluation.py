from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score,roc_auc_score,precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def eval(predict,Y_test):
    acc = get_accuracy(predict, Y_test)
    auc = get_auc(predict, Y_test)
    f1 = get_f1(predict, Y_test)
    print("acc: " + str(acc) + ' auc: ' + str(auc) + 'f1 ' + str(f1))
    return acc,auc,f1


# accuracy, both parameters are numpy type
def get_accuracy(predict, Y):
    correct = 0
    N = len(Y)
    for i in range(N):
        if predict[i] == Y[i]:
            correct += 1
    return correct /N


def get_auc(predict,Y):
    return roc_auc_score(Y,predict)


def get_f1(predict,Y):
    return f1_score(Y, predict, average='binary')


def cor_heatmap(train):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)


def draw_pr(predict, Y):
    precision, recall, thresholds = precision_recall_curve(Y, predict)
    plt.plot(recall, precision)
    plt.show(block=True)


def draw_feature_importance(feature_importance):
    feature = range(np.size(feature_importance))
    featureImportance = np.vstack((feature, feature_importance))
    df = pd.DataFrame(featureImportance)
    sns.barplot(data=df)
    plt.show()


def drawAUC(predict,Y):
    # fpr, tpr, thresholds = roc_curve(Y, predict, pos_label=2)
    # # Draw the ROC plot
    # plt.figure()
    # lw = 2
    # plt.figure(figsize=(10, 10))
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % aucNumber)  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC')
    # plt.legend(loc="lower right")
    # plt.show()
    fpr, tpr, threshold = roc_curve(Y, predict)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def eval_with_plot(predict,label):
    # accuracy
    print('acc ')
    acc = get_accuracy(predict,label) #accuracy_score(label, predict)
    print(acc)

    # F1
    print("f1")
    f1=get_f1(predict,label)
    print(f1)

    print('auc')
    auc_cur = get_auc(predict,label)
    print(auc_cur)
    draw_pr(predict, label)


    print("auc")
    fpr, tpr, thresholds = roc_curve(label, predict)
    ##print(fpr, '\n', tpr, '\n', thresholds)
    aucNumber = auc(fpr, tpr)
    print(aucNumber)

    # Draw the ROC plot
    lw=2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % aucNumber)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show(block=True)
    return acc,auc_cur,f1
