import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_training import raw_prediction, smooth, build_model, masks_as_color, multi_rle_encode, DiceLoss, dice_coef
from skimage.io import imread
from tensorflow.keras.optimizers import Adam

WEIGHTS_PATH = './seg_model_weights.best.hdf5'
PATH = './airbus-ship-detection/test_v2/'
MODEL_SHAPE = (384, 384, 3)


def pred_encode(model, img, path=PATH, **kwargs):
    cur_seg, _ = raw_prediction(model, img, path=path)
    cur_seg = smooth(cur_seg)

    cur_rles = multi_rle_encode(cur_seg, **kwargs)
    return [[img, rle] for rle in cur_rles if rle is not None]


if __name__ == '__main__':
    images_paths = np.array(os.listdir(PATH))
    model = build_model(input_shape=MODEL_SHAPE)
    model.load_weights(WEIGHTS_PATH)
    model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=DiceLoss(), metrics=[dice_coef])

    out_pred_rows = []
    for c_img_name in images_paths[:100]:
        out_pred_rows += pred_encode(model, c_img_name, min_max_threshold=1.0)

    df = pd.DataFrame(out_pred_rows)
    df.columns = ['ImageId', 'EncodedPixels']
    df = df[df.EncodedPixels.notnull()]

    PREDICTIONS_NUM = 5

    fig, m_axs = plt.subplots(PREDICTIONS_NUM, 2, figsize=(9, PREDICTIONS_NUM * 5))
    [c_ax.axis('off') for c_ax in m_axs.flatten()]

    for (ax1, ax2), c_img_name in zip(m_axs, np.random.choice(pd.unique(df["ImageId"]), PREDICTIONS_NUM)):
        c_img = imread(os.path.join(PATH, c_img_name))
        c_img = np.expand_dims(c_img, 0) / 255.0
        ax1.imshow(c_img[0])
        ax1.set_title('Image: ' + c_img_name)
        ax2.imshow(masks_as_color(df[df['ImageId'] == c_img_name]['EncodedPixels'], shape=MODEL_SHAPE[:2]))
        ax2.set_title('Prediction')

    fig.savefig('predictions.png')
