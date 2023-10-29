import numpy as np
from PIL import Image


def upper_int(x):
    print(x)
    if x < 0:
        return 0
    if x > int(x):
        return int(x) + 1
    else:
        return int(x)


def same_padding2(kernel_size, stride, height, width, dilation=1):
    padding = upper_int(((stride - 1) * height - stride + dilation * kernel_size - dilation + 1) / 2), upper_int(
        ((stride - 1) * width - stride + dilation * kernel_size - dilation + 1) / 2)
    return padding


def same_padding(kernel_size, stride, dilation=1):
    padding = upper_int((dilation * kernel_size - dilation + 1 - stride) / 2)
    return padding


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0


def save_images(filepath, result_1, result_2=None):
    result_1 = result_1.cpu().detach().numpy()
    if result_2 != None:
        result_2 = result_2.cpu().detach().numpy()
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)
    if len(result_1.shape) == 3:
        result_1 = result_1.transpose(1, 2, 0)
    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis=1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')
