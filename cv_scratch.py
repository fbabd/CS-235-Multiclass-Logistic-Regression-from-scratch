import numpy as np
import pandas as pd
from MyLogisticRegression import LR_Multiclass


def print_cls_report(y_true, y_pred):
    """
    y_true      :   Actual Labels
    y_pred      :   Predicted Labels 
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    # target_names contains the name of each label
    target_names = ['#people = 0', '#people = 1', '#people = 2', '#people = 3']
    # Following function calculates the classification performances
    cr = classification_report(y_true, y_pred, target_names = target_names, output_dict=False)
    # Following function calculates the confusion matrix of classification 
    cm = confusion_matrix(y_true, y_pred)
    print(cr)
    sns.set(font_scale=1)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 12}, cmap="Blues") # font size
    plt.show()


def run_model(X_train, X_test, y_train, y_test, regu='l2'):
    """
    X_train     :   Train set data
    y_train     :   Train set label
    X_test      :   Test set data
    y_test      :   Test set label 
    regu        :   Regularization. Value can be 'l1' or 'l2'. Default is 'l2'. 
    """
    clf = LR_Multiclass(regularization=regu)
    clf.fit(np.array(X_train), np.array(y_train)) 
    clf.plot_history()
    y_pred = clf.predict(X_test)
    print_cls_report(y_test, y_pred)



def cv_test(X_, y_, k=3, regu='l2'):
    """ 
    X__         :   Features of the data
    y__         :   Lables of the data
    k           :   number of fold
    regu        :   'l1' or 'l2' for the Logisitic Regression model. Default is 'l2'
    """
    import random
    random.seed(123)
    from sklearn.metrics import classification_report

    # Following section finds all unique labels and count of each label in the dataset.
    """  
    labels      :   unique labels from y__
    lb_freq     :   counts for each unique labels 
    indices     :   original index in the given dataset for each label 
    """
    labels, lb_freq = np.unique(y_, return_counts=True)
    indices = []
    for lb in labels:
        ids = np.where(y_ == lb)
        indices.append( list(ids[0]))

    """
    accuracy    :   accuracy scores in all folds
    f1          :   f1 scores in all folds
    precision   :   precison scores in all folds
    recall      :   recall scores in all folds
    """
    accuracy = []
    f1 = []
    precision = []
    recall = []

    # Following code section prints the amount of data from each label given as input
    # print("\n ====== Total Data ", "  ========= ")
    # for i in range(len(labels)):
    #     print( "class = ", labels[i], " #sample = ", lb_freq[i], end=" , ")

    # Following code section creates data by taking 1 fold as test-set and rest (k-1) set as train-set and runs the classifier
    """
    this_fold   :   contains the index numbers for the test set in current fold 
    X_test_     :   test set data for the current fold
    y_test_     :   test set labels for the current fold
    X_train_    :   train set data for the current fold
    y_train_    :   train set labels for the current fold
    
    """
    for i in range(k):
        print("Fold : ", i+1, " => ", end=" ")
        # Following section finds the random indices for the test set of this fold
        this_fold = []
        for cls in range(len(lb_freq)):
            fold_size = int(lb_freq[cls]/k) # finding the fold size class wise 
            # print( "class = ", cls, "#sample = ", fold_size, end=" , ") # print the class wise fold data 
            # Following loop, randomly select an index for this class without replacement, and append to this_fold 
            count = 0
            while(count < fold_size):  
                idx = random.randrange(len(indices[cls])) 
                t_id = indices[cls][idx]
                this_fold.append(t_id)
                indices[cls] = np.delete(indices[cls], idx)
                count+=1
        #print()

        # Following section creates the train and test set for classification
        X_test_ = np.array(X_)[this_fold]
        y_test_ = np.array(y_)[this_fold]
        print("test size: ", X_test_.shape, y_test_.shape, end=" | ")
        a = list(range(X_.shape[0]))
        train_idx =  [x for x in a if x not in this_fold] 
        X_train_ = np.array(X_)[train_idx]
        y_train_ = np.array(y_)[train_idx]
        print("train size: ", X_train_.shape, y_train_.shape)

        # Following section runs the classifier and calculates the performance metrics
        clf = LR_Multiclass(regularization=regu)
        clf.fit(X_train_, y_train_) 
        y_pred_ = clf.predict(X_test_)
        target_names = ['#people = 0', '#people = 1', '#people = 2', '#people = 3']
        m = classification_report(y_test_, y_pred_, target_names = target_names, output_dict=True)
        accuracy.append(m['accuracy'])
        f1.append(m['macro avg']['f1-score'])
        precision.append(m['macro avg']['precision'])
        recall.append(m['macro avg']['recall'])
    

    # Following dictionary contains all the scores of the folds
    """ 
    all_scores  :   all score of the folds in a dictionary
    mean_scores :   mean accuracy, f1, precision, and recall
    err_scores  :   standard error of the mean scores    
    """ 
    all_scores = {
        "acc": accuracy,
        "f1": f1,
        "prc": precision,
        "rec": recall
    }
    
    mean_scores = [np.mean(np.array(accuracy)), np.mean(np.array(f1)), np.mean(np.array(precision)), np.mean(np.array(recall))]
    
    err_scores = [  np.std(np.array(accuracy))/np.sqrt(len(accuracy)), 
                    np.std(np.array(f1))/np.sqrt(len(f1)),
                    np.std(np.array(precision))/np.sqrt(len(precision)), 
                    np.std(np.array(recall))/np.sqrt(len(recall)) ]
    #print(mean_scores)
    print_cv_scores(all_scores, mean_scores, err_scores)
    plot_cv_results(all_scores, mean_scores, err_scores)
    return all_scores, mean_scores, err_scores
    


def plot_cv_results(all_scores, mean_scores, error_scores):
    """
    all_scores  :   all score of the folds in a dictionary
    mean_scores :   mean accuracy, f1, precision, and recall
    error_scores  :   standard error of the mean scores 
    """
    import matplotlib.pyplot as plt 
    
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(all_scores['acc'])), all_scores['acc'], label="Accuracy")
    ax2.plot(range(len(all_scores['f1'])), all_scores['f1'], label="F1 Score")
    ax2.plot(range(len(all_scores['prc'])), all_scores['prc'], label="Precision")
    ax2.plot(range(len(all_scores['rec'])), all_scores['rec'], label="Recall")
    ax2.legend()
    ax2.set_title('Performance Scores in each Fold')
    ax2.set_xlabel(" Fold No. ")
    ax2.set_ylabel("Percentage (%) ")
    ax2.yaxis.grid(True)
    ax2.xaxis.grid(True)
    
    fig, ax = plt.subplots()
    x_pos = range(len(mean_scores))
    ax.bar(x_pos,mean_scores, yerr=error_scores, align='center', alpha=0.65, ecolor='black', capsize=5)
    ax.set_ylim(0.8, 1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Accuracy", "F1 Score", "Precision", "Recall"])
    ax.set_title('Mean performance scores with standard error \n')
    ax.set_xlabel("Evaluation Metric \n")
    ax.set_ylabel("Percentage (%) ")
    ax.yaxis.grid(True)


def print_cv_scores(all_scores, mean_scores, error_scores):
    print("Crosss-Validation Results: ")
    print("Metric   ", "\t", "Mean ",  "\t\t\t", "Error")
    print("Accuracy ", "\t ", mean_scores[0], "\t", error_scores[0])
    print("F1-score ", "\t ", mean_scores[1], "\t", error_scores[1])
    print("Precision", "\t ", mean_scores[2], "\t", error_scores[2])
    print("Recall   ", "\t ", mean_scores[3], "\t", error_scores[3])
    print("All Scores: ")
    df = pd.DataFrame(all_scores) 
    print(df)
