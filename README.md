# targeted_UAP_CIFAR10

学習済みモデルは、[GuanqiaoDing/CNN-CIFAR10](https://github.com/GuanqiaoDing/CNN-CIFAR10)を使用。

1. Targeted UAPをインストール

    `pip install git+https://github.com/hkthirano/adversarial-robustness-toolbox`

1. 実行

    ```sh
    python make_noise.py

    # norm2: 6 %
    # acc_train: 97 %
    # acc_test: 91 %
    # acc_train_adv: 18 %
    # acc_test_adv: 18 %
    # targeted_success_rate_train: 87 %
    # targeted_success_rate_test: 87 %
    ```
