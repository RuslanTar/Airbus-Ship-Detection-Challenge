# Kaggle: [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)

## Overview

### System
OS: Windows 11 21H2

CPU: Intel Core i5-8400

GPU: Nvidia GTX 1080 Ti 11GB

RAM: 16 GB (+32GB virtual RAM)

### Data

The data is available on the [Kaggle website](https://www.kaggle.com/competitions/airbus-ship-detection/data).

You can download it in folder

+ airbus-ship-detection/

  + test_v2 

  + train_v2

### Data augmentation

The data for training contains 192,556 768*768 images. I use a module called ImageDataGenerator in keras.preprocessing.image to do data augmentation.

### Model

<img src="./imgs/u-net architecture.png">

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Sigmoid activation function makes sure that mask pixels are in [0, 1] range.

### Training

The model train up to 10 passes of 100 epochs (with early stopping callbacks).

After training, calculated dice coefficient is about 0.47.

Loss function for the training is dice loss function.

## How to use

### Dependencies

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)
2. [Install Tensorflow](https://www.tensorflow.org/install)
3. Install dependencies

`pip install -r requirements.txt`

This project depends on the following libraries:

- Tensorflow: v2.9.1
- Tensorflow_gpu: v2.9.1
- Keras: v2.9.0
- Matplotlib: v3.5.3
- NumPy: v1.23.0
- Pandas: v1.4.3
- scikit_image: v0.19.2
- scikit_learn: v1.1.2

Also, this code should be compatible with Python versions 3.7–3.10.

### Train model

```console
> python model_training.py -h
usage: model_training.py [-h] [--epochs E] [--steps STEPS] [--batch-size B] [--load LOAD] [--validation VAL]
                         [--net-scale NET_SCALING] [--img-scale IMG_SCALING] [--val-imgs VALID_IMG_COUNT]
                         [--train TRAIN] [--test TEST] [--segmentation SEGMENTATION]

Train the UNet on images and target masks

options:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --steps STEPS, -t STEPS
                        Maximum number of steps_per_epoch in training
  --batch-size B, -b B  Batch size
  --load LOAD, -f LOAD  Load model weights from a .hdf5 file
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --net-scale NET_SCALING, -n NET_SCALING
                        Downsampling inside the network
  --img-scale IMG_SCALING, -s IMG_SCALING
                        Downsampling in preprocessing
  --val-imgs VALID_IMG_COUNT, -i VALID_IMG_COUNT
                        Number of validation images to use
  --train TRAIN         Path to train folder
  --test TEST           Path to test folder
  --segmentation SEGMENTATION
                        Path to train_ship_segmentations_v2.csv file
```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

### Inference model

After training your model and saving its weights to `seg_model_weights.best.hdf5`, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python model_inference.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python model_inference.py -i image1.jpg image2.jpg --show --no-save`

```console
> python predict.py -h
usage: model_inference.py [-h] [--weights FILE] --input INPUT [INPUT ...] [--output OUTPUT [OUTPUT ...]] [--show]
                          [--no-save] [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

options:
  -h, --help            show this help message and exit
  --weights FILE, -w FILE
                        Specify the file in which the model weights are stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images in current directory
  --output OUTPUT [OUTPUT ...], -o OUTPUT [OUTPUT ...]
                        Filenames of output images
  --show, -v            Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```

You can specify which weights file to use with `--weights seg_model_weights.best.hdf5`.

## Results

The model copes well with pictures with good contrast and where there are large obvious ships. Also when there are no shores of piers in the picture, etc.

On the other hand, in pictures where there are a lot of small ships, the model copes poorly. Also, in cases where several ships are nearby, it considers it as one object.

Probably, in order to improve the result, it is worth learning the model on full-size pictures. It is even possible to start learning on lower resolution pictures to reinforce the generalization ability, and then on higher resolution to improve the accuracy of predictions.

![](./imgs/validation.png)
