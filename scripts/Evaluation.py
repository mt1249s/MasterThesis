def eval_metrics(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    from sklearn.metrics import mean_squared_error, accuracy_score
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    #print(f'accuracy:{accuracy:.4f}')
    percision = TP / (TP + FP)
    #print(f'precision:{percision:.4f}')
    Recall = TP / (TP + FN)
    #print(f'Recall:{Recall:.4f}')
    Specificity = TN / (TN + FP)
    #print(f'Specificity:{Specificity:.4f}')
    F1_score = 2 * (percision * Recall) / (percision + Recall)
    #print(f'F1_score:{F1_score:.4f}')

    # return(TP/(TP+TN), FP/(FP+FN), TN/(TP+TN), FN/(FN+FP))
    return (TP, FP, TN, FN, accuracy, percision, Recall, Specificity, F1_score)