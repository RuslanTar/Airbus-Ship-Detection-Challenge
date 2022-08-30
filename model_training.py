import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from skimage.io import imread
from skimage.morphology import binary_opening, disk, label
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import gc

gc.enable()  # memory is tight

# corrupted images
exclude_list = ['6384c3e78.jpg']

args = None


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--steps', '-t', type=int, default=30, help='Maximum number of steps_per_epoch in training')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model weights from a .hdf5 file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--net-scale', '-n', dest='net_scaling', type=float, default=1,
                        help='Downsampling inside the network')
    parser.add_argument('--img-scale', '-s', dest='img_scaling', type=float, default=0.5,
                        help='Downsampling in preprocessing')
    parser.add_argument('--val-imgs', '-i', dest='valid_img_count', type=int, default=1500,
                        help='Number of validation images to use')
    parser.add_argument('--train', type=str, default='./airbus-ship-detection/train_v2/', help='Path to train folder')
    parser.add_argument('--test', type=str, default='./airbus-ship-detection/test_v2/', help='Path to test folder')
    parser.add_argument('--segmentation', type=str, default='./airbus-ship-detection/train_ship_segmentations_v2.csv',
                        help='Path to train_ship_segmentations_v2.csv file')

    return parser.parse_args()


# RLE (Run-Length Encode and Decode)
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode


def multi_rle_encode(img, **kwargs):
    """
    Encode connected regions as separated masks
    """
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels == k, axis=2), **kwargs) for k in np.unique(labels[labels > 0])]
    else:
        return [rle_encode(labels == k, **kwargs) for k in np.unique(labels[labels > 0])]


def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    if np.max(img) < min_max_threshold:
        return ''  # no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return ''  # ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


def masks_as_color(in_mask_list, shape=(768, 768)):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros(shape, dtype=np.float32)
    scale = lambda x: (len(in_mask_list) + x + 1) / (len(in_mask_list) * 2)  # scale the heatmap image to shift
    for i, mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:, :] += scale(i) * rle_decode(mask, shape=shape)
    return all_masks


# Split into training and validation groups

def make_image_gen(in_df, path, batch_size, img_scaling):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(path, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            if img_scaling is not None:
                c_img = c_img[::img_scaling[0], ::img_scaling[1]]
                c_mask = c_mask[::img_scaling[0], ::img_scaling[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0) / 1.0
                out_rgb, out_mask = [], []


def create_dataframes(segmentation_path, train_path, batch_size, test_size, valid_img_count, img_scaling):
    masks = pd.read_csv(segmentation_path)

    # Drop all corrupted images
    masks.drop(index=masks[masks["ImageId"].isin(exclude_list)].index, inplace=True)

    # Getting unique images ids
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    masks.drop(['ships'], axis=1, inplace=True)

    # Undersample Empty Images
    samples_per_group = 2000
    balanced_train_df = unique_img_ids \
        .groupby('ships') \
        .apply(lambda x: x.sample(samples_per_group) if len(x) > samples_per_group else x)
    balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max() + 1)

    train_ids, valid_ids = train_test_split(balanced_train_df,
                                            test_size=test_size / 100,
                                            stratify=balanced_train_df['ships'])
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids)

    train_x, train_y = next(make_image_gen(train_df, train_path, batch_size, img_scaling))

    valid_x, valid_y = next(make_image_gen(valid_df, train_path, valid_img_count, img_scaling))
    gc.collect()

    return train_df, train_x, train_y, valid_x, valid_y


# Augment Data
def setup_augment_params(featurewise_center=False, samplewise_center=False, rotation_range=45, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.01, zoom_range=[0.9, 1.25], horizontal_flip=True,
                         vertical_flip=True, fill_mode='reflect', data_format='channels_last'):
    dg_args = dict(featurewise_center=featurewise_center,
                   samplewise_center=samplewise_center,
                   rotation_range=rotation_range,
                   width_shift_range=width_shift_range,
                   height_shift_range=height_shift_range,
                   shear_range=shear_range,
                   zoom_range=zoom_range,
                   horizontal_flip=horizontal_flip,
                   vertical_flip=vertical_flip,
                   fill_mode=fill_mode,
                   data_format=data_format)
    global image_gen, label_gen

    image_gen = ImageDataGenerator(**dg_args)
    label_gen = ImageDataGenerator(**dg_args)


def create_aug_gen(in_gen, seed=42, **kwargs):
    setup_augment_params(**kwargs)
    np.random.seed(seed)
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255 * in_x,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)

        yield next(g_x) / 255.0, next(g_y)


# Build U-Net model
def build_model(net_scaling, input_shape=(768, 768)):

    def upsample_simple(filters, kernel_size, strides, padding):
        return layers.UpSampling2D(strides)

    upsample = upsample_simple

    input_img = layers.Input(input_shape, name='RGB_Input')
    pp_in_layer = input_img

    if net_scaling is not None:
        pp_in_layer = layers.AvgPool2D(net_scaling)(pp_in_layer)

    pp_in_layer = layers.GaussianNoise(0.1)(pp_in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(pp_in_layer)
    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = upsample(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = upsample(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = upsample(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = upsample(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    # d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    # d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
    if net_scaling is not None:
        d = layers.UpSampling2D(net_scaling)(d)

    seg_model = models.Model(inputs=[input_img], outputs=[d])
    return seg_model


# Train model

# Intersection over Union
def IoU(y_true, y_pred, eps=1e-6):
    # if K.max(y_true) == 0.0:
    #     return IoU(1-y_true, 1-y_pred)  # empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return -K.mean((intersection + eps) / (union + eps), axis=0)


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result


def dice_coef(mask1, mask2):
    intersect = K.sum(mask1 * mask2)
    fsum = K.sum(mask1)
    ssum = K.sum(mask2)
    dice = (2 * intersect) / (fsum + ssum)
    dice = K.mean(dice)
    return dice


def get_callbacks_list():
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

    checkpoint = ModelCheckpoint('./seg_model_weights.best.hdf5', monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                       patience=1, verbose=1, mode='min',
                                       min_delta=0.0001, cooldown=0, min_lr=1e-8)

    early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
                          patience=20)

    callbacks_list = [checkpoint, early, reduceLROnPlat]
    return callbacks_list


def fit(model, batch_size, epochs, steps, train_path, train_df, valid_x, valid_y, loss_func, metrics, callbacks_list):
    model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=loss_func, metrics=metrics)

    step_count = min(steps, train_df.shape[0] // batch_size)
    aug_gen = create_aug_gen(make_image_gen(train_df, train_path, batch_size, img_scaling))
    loss_history = [model.fit(aug_gen,
                              steps_per_epoch=step_count,
                              epochs=epochs,
                              validation_data=(valid_x, valid_y),
                              callbacks=callbacks_list,
                              workers=1  # the generator is not thread safe
                              )]
    return loss_history


def raw_prediction(model, c_img_name, path, img_scaling=None):
    c_img = imread(os.path.join(path, c_img_name))
    c_img = np.expand_dims(c_img, 0) / 255.0
    if img_scaling is not None:
        c_img = c_img[:, ::img_scaling[0], ::img_scaling[1]]
    cur_seg = model.predict(c_img)[0]
    return cur_seg, c_img[0]


def smooth(cur_seg):
    return binary_opening(cur_seg > 0.99, np.expand_dims(disk(2), -1))


def valid_image(model, valid_df, train_path, img_scaling):
    # Get a sample of each group of ship count
    samples = valid_df.groupby('ships').apply(lambda x: x.sample(1))
    fig, m_axs = plt.subplots(samples.shape[0], 4, figsize=(15, samples.shape[0] * 4))
    [c_ax.axis('off') for c_ax in m_axs.flatten()]

    for (ax1, ax2, ax3, ax4), c_img_name in zip(m_axs, samples.ImageId.values):
        first_seg, first_img = raw_prediction(model, c_img_name, train_path, img_scaling)
        ax1.imshow(first_img)
        ax1.set_title('Image: ' + c_img_name)
        ax2.imshow(first_seg[:, :, 0], cmap=get_cmap('jet'))
        ax2.set_title('Model Prediction')
        reencoded = masks_as_color(multi_rle_encode(smooth(first_seg)[:, :, 0]), shape=first_seg.shape[:2])
        ax3.imshow(reencoded)
        ax3.set_title('Prediction Masks')
        ground_truth = masks_as_color(valid_df[valid_df['ImageId'] == c_img_name]['EncodedPixels'])
        ax4.imshow(ground_truth)
        ax4.set_title('Ground Truth')

    fig.savefig('validation.png')


def DICE_COEF(mask1, mask2):
    intersect = np.sum(mask1 * mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3)  # for easy reading
    return dice


def scale_to_tuple(scale):
    scale = int(round(1 / scale))
    return (scale, scale)

if __name__ == '__main__':
    args = get_args()

    # downsampling in preprocessing
    img_scaling = scale_to_tuple(args.img_scaling)
    net_scaling = scale_to_tuple(args.net_scaling)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    logging.info("Reading data ...")
    train_df, train_x, train_y, valid_x, valid_y = create_dataframes(args.segmentation, args.train, args.batch_size,
                                                                     args.val, args.valid_img_count, img_scaling)

    logging.info("Building model ...")
    seg_model = build_model(args.net_scaling, input_shape=train_x.shape[1:])
    loss_func = DiceLoss()
    metrics = [IoU, 'binary_accuracy', dice_coef]
    callbacks_list = get_callbacks_list()
    if args.load:
        seg_model.load_weights(args.load)
        seg_model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=loss_func, metrics=metrics)
        logging.info(f'Model loaded from {args.load}')
        
    else:
        logging.info(f'''Starting training:
                Epochs:          {args.epochs}
                Batch size:      {args.batch_size}
                Steps per epoch: {args.steps}
                Training size:   {train_df.shape[0]}
                Validation size: {valid_x.shape[0]}
                Images scaling:  {args.img_scaling}
            ''')

        pass_num = 1
        while True:
            loss_history = fit(seg_model, args.batch_size, args.epochs, args.steps, args.train, train_df,
                               valid_x, valid_y, loss_func, metrics, callbacks_list)

            if np.min([mh.history['val_loss'] for mh in loss_history]) < 0.2 or pass_num >= 10:
                break
            pass_num += 1

        # seg_model.save('seg_model.h5')
        gc.collect()

    y_pred = seg_model.predict(valid_x)
    logging.info(f"Dice score: {DICE_COEF(valid_y, y_pred)}")
