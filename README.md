# NanoGPTForge

![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**NanoGPTForge** is a modified version of @karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) that focuses on **simplicity, clean code, type safety**, and leverages standard PyTorch tools. It is designed to be **plug-and-play**, allowing fast experimentation with minimal setup.

## Features

- Plug-and-play architecture for rapid experimentation
- Type-safe, clean, and readable code
- Uses common PyTorch components for maintainability and compatibility
- Lightweight and minimalistic implementation
- Easily extendable to new datasets, models, or training scripts

## Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

This repository supports the [enwik8 dataset](http://prize.hutter1.net/). To prepare the dataset:

```bash
python data/enwik8/prepare.py
```

This will encode the text into tokens, ready for training.

## Training

Check training script parameters:

```bash
python train.py --help
```

Example usage:

```bash
python train.py --model_name nanogpt --dataset_name enwik8
```

## Sampling / Inference

Check inference script aprameters

```bash
python sample.py --help
```

Example usage:

```bash
python sample.py --checkpoint_file_name nanogpt-100 --prompt "How to build a DIY drone?"
```

## Why NanoGPTForge?

1. Plug-and-play setup for easy experimentation
2. Lightweight, minimalistic implementation
3. Type-safe code reduces runtime errors
4. Clean and readable codebase
5. Built on the trusted nanoGPT foundation
6. Extensible to new datasets and architectures
7. Fully PyTorch-native
8. Works on CPU and GPU without complex configuration
9. Ideal for educational purposes and experimentation
10. Focused on simplicity without sacrificing flexibility

## License

This project is based on @karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). Please check the original license in [nanoGPT](https://github.com/karpathy/nanoGPT) for details.  
Licensed under the [MIT License](https://opensource.org/licenses/MIT).
