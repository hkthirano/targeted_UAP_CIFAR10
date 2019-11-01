# targeted_UAP_CIFAR10

We used the VGG-20 model for the CIFAR-10 dataset obtained from a GitHub repository [GuanqiaoDing/CNN-CIFAR10](https://github.com/GuanqiaoDing/CNN-CIFAR10)

1. Install Targeted UAP

    `pip install git+https://github.com/hkthirano/adversarial-robustness-toolbox`

1. Make noise

    ```sh
    python make_noise.py
    
    # === Targeted UAP ===
    # norm2: 4.8 %
    # targeted_success_rate_train: 79.4 %
    # targeted_success_rate_test: 79.0 %
    # === Random Noise ===
    # norm2_rand: 4.8 %
    # targeted_success_rate_train_rand: 9.7 %
    # targeted_success_rate_test_rand: 9.7 %
    ```
