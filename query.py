# -*- coding: utf-8 -*-
import numpy as np
import h5py

from FeatureExtractor.cnn_based_extractor import VGGNet

import logging.config
LOG = logging.getLogger("")


def query(query_image_path, index_file, uniform_size=(224, 224, 3), limit=10):
    # read in indexed images' name and feature vectors
    h5f = h5py.File(index_file, 'r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()

    LOG.debug("Start Searching")

    # init VGGNet Model and extract the query image's feature
    model = VGGNet(input_shape=uniform_size)
    queryDir = query_image_path
    queryVec = model.extract_feat(queryDir)
    # Cosine Similarity
    scores = np.dot(queryVec, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    # top retrieved images to show
    imlist = [str(imgNames[index], 'utf-8') for i, index in enumerate(rank_ID[0:limit])]
    LOG.debug("top %d images in order are: " % limit, imlist)
    return imlist
