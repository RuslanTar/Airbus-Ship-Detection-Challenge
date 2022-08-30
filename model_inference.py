import os
import argparse
import logging
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from model_training import raw_prediction, smooth, build_model, masks_as_color, multi_rle_encode, DiceLoss, dice_coef, scale_to_tuple, masks_as_color
from skimage.io import imread
from tensorflow.keras.optimizers import Adam

WEIGHTS_PATH = './seg_model_weights.best.hdf5'
PATH = './airbus-ship-detection/test_v2/'
MODEL_SHAPE = (384, 384, 3)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--weights', '-w', default='./seg_model_weights.best.hdf5', metavar='FILE',
                        help='Specify the file in which the model weights are stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='Filenames of input images in current directory', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--show', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    return parser.parse_args()


def pred_encode(model, img, path, **kwargs):
    cur_seg, _ = raw_prediction(model, img, path=path)
    cur_seg = smooth(cur_seg)

    cur_rles = multi_rle_encode(cur_seg, **kwargs)
    return [[img, rle] for rle in cur_rles if rle is not None]


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    model_shape = (int(768*args.scale), int(768*args.scale), 3)
    img_scaling = scale_to_tuple(args.scale)

    # images_paths = np.array(os.listdir(args))
    model = build_model((1, 1), input_shape=model_shape)
    model.load_weights(args.weights)
    model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=DiceLoss(), metrics=[dice_coef])

    for i, c_img_name in enumerate(in_files):
        logging.info(f'\nPredicting image {c_img_name} ...')
        mask, _ = raw_prediction(model, c_img_name, path=os.getcwd(), img_scaling=img_scaling)
        mask = masks_as_color(multi_rle_encode(mask, min_max_threshold=1.0), shape=model_shape[:2])

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.show:
            img = Image.open(c_img_name)
            logging.info(f'Visualizing results for image {c_img_name}, close to continue...')
            plot_img_and_mask(img, mask)
