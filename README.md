# Targeted universal adversarial perturbation
This repository contains Keras implementation of our simple iterative method for generating a targeted universal adversarial perturbation (UAP), which causes deep natural networks to classify most input images into a specific class, as described in the following paper:

Hirano H and Takemoto K (2020) **Simple iterative method for generating targeted universal adversarial perturbations.** in Proceedings of 25th International Symposium on Artificial Life and Robotics (AROB 25th 2020), *to appear*. [arXiv:1911.06502](https://arxiv.org/abs/1911.06502)

As a demo, we used the VGG-20 model for the CIFAR-10 dataset obtained from a GitHub repository [GuanqiaoDing/CNN-CIFAR10](https://github.com/GuanqiaoDing/CNN-CIFAR10)

## Usage
1. Install the targeted UAP method.

    `pip install git+https://github.com/hkthirano/adversarial-robustness-toolbox`

1. Generate a targeted UAP.

    ```sh
    python generate_noise.py
    
    # === Targeted UAP ===
    # norm2: 4.8 %
    # targeted_success_rate_train: 79.4 %
    # targeted_success_rate_test: 79.0 %
    # === Random Noise ===
    # norm2_rand: 4.8 %
    # targeted_success_rate_train_rand: 9.7 %
    # targeted_success_rate_test_rand: 9.7 %
    ```
