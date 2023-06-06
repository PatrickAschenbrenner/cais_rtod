import cv2


def hog_descriptor() -> cv2.HOGDescriptor:
    """
    Get the HOG descriptor with the given window size (width, height).
    Assuming an image resolution of size 256x256.
    :return:         HOG descriptor
    """
    win_size = (256, 256)
    block_size = (256, 256)
    block_stride = (128, 128)
    cell_size = (128, 128)
    nbins = 9
    deriv_aperture = 1
    win_sigma = -1
    histogram_norm_type = 0
    l2_hys_threshold = 0.2
    gamma_correction = 1
    nlevels = 64
    hog = cv2.HOGDescriptor(win_size,block_size,block_stride,cell_size,nbins,deriv_aperture,win_sigma,
                            histogram_norm_type,l2_hys_threshold,gamma_correction,nlevels)
    return hog


def get_img_feature(image,
                    hog: cv2.HOGDescriptor,
                    source: str = 'file'):
    """
    Compute the HOG feature vector of the image.
    :param image:      image location or image object
    :param hog:        HOG descriptor
    :param source:     type of image (file or array)
    :return:           HOG feature vector
    """
    if source == 'file':
        img = cv2.imread(image)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif source == 'array':
        # Assume a BGR colour image
        img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError('Image type not supported')
    img_hog = hog.compute(img_grey)
    feature_vector = img_hog.flatten()
    return feature_vector
