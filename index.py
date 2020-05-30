# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np

from FeatureExtractor.cnn_based_extractor import VGGNet

import logging.config
LOG = logging.getLogger("")

def get_imlist(path):
    '''
    Returns a list of filenames for all jpg images in a directory.
    :param path: absolute gallery dataset paths
    :return: Return a list of absolute image paths
    '''
    LOG.debug("%d images in total." % len(os.listdir(path)))
    # img_list = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    img_list = [os.path.join(path, f) for f in os.listdir(path)]
    return img_list


def index_images(gallery_database_path, index_file, uniform_size=(224, 224, 3)):
    '''
    Extract features for gallery database and store
    :param gallery_database_path: absolute path
    :param index_file: absolute .h5 path
    :param uniform_size: image shape for extracting feature
    :return:
    '''
    db = gallery_database_path
    # TODO: Cannot read all directory all in once
    img_list = get_imlist(db)
    LOG.debug('feature extraction starts...')

    feats = []
    names = []

    model = VGGNet(input_shape=uniform_size)
    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path=img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        LOG.debug("extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list)))
    feats = np.array(feats)

    # directory for storing extracted features
    output = index_file
    LOG.debug('writing feature extraction results ...')

    # TODO: 'a' or 'w' for .h5
    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=np.string_(names))
    h5f.close()
    return index_file