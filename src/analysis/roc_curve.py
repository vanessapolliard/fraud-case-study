from sklearn.metrics import roc_curve, auc

def plot_roc_auc(X,y):
    fpr, tpr, thresholds = roc_curve(y, pipeline.set_params().predict_proba(X)[:,1])
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == '__main__':
    y_test = 
    X_test = 
