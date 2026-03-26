from src.thresholding import compute_confusion_matrix

def test_confusion_matrix():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]

    conf = compute_confusion_matrix(y_true, y_pred)

    assert conf["tp"] == 1
    assert conf["tn"] == 2
    assert conf["fp"] == 0
    assert conf["fn"] == 1