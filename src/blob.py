import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def load_data(
    n_samples=1000, n_features=2, n_classes=4, random_state=42, train_size=0.2
):
    """
    多クラスデータセットを生成する

    Args:
        n_samples:      サンプル数
        n_features:     データの変数の数（n_features=2ならxとy）
        n_classes:      クラスタの数
        random_state:   乱数のシード値
        train_size:     訓練データの割合
    Returns:
        X_blob_train:   訓練用入力データ
        X_blob_test:    テスト用入力データ
        y_blob_train:   訓練用正解ラベルデータ
        y_blob_test:    テスト用正解ラベルデータ
    """

    # データの生成
    X_blob, y_blob = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=1.5,
        random_state=random_state,
    )

    # Tensorに変換
    X_blob = torch.from_numpy(X_blob).type(torch.float)
    y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

    # 訓練データとテストデータに分割
    X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
        X_blob, y_blob, test_size=train_size, random_state=random_state
    )

    return (X_blob_train, X_blob_test, y_blob_train, y_blob_test)
