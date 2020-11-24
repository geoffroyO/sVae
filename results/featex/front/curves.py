import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    res = pd.read_csv("./model_history_log.csv")
    res = res.drop(0)

    f1score = []
    val_f1score = []
    for index, row in res.iterrows():
        precision, recall = row["precision"], row["recall"]
        f1 = 2 * precision * recall / (precision + recall)
        f1score.append(f1)

        val_precision, val_recall = row["val_precision"], row["val_recall"]
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall)
        val_f1score.append(val_f1)

    res["f1"] = f1score
    res["val_f1"] = val_f1score

    fig = plt.figure()
    plt.plot(res['accuracy'])
    plt.plot(res['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./accuracy")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(res['auc'])
    plt.plot(res['val_auc'])
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./auc")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(res['loss'])
    plt.plot(res['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./loss")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(res['precision'])
    plt.plot(res['val_precision'])
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./precision")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(res['recall'])
    plt.plot(res['val_recall'])
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./recall")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(res['f1'])
    plt.plot(res['val_f1'])
    plt.title('model f1-score')
    plt.ylabel('f1-score')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./f1_score")
    plt.close(fig)