# -*- coding: utf-8 -*-
import os
import time
from functools import wraps
import cv2
import numpy as np

import index
import query

import logging.config
LOG = logging.getLogger("")

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.__name__ , str(t1 - t0))
              )
        LOG.debug("Total time running %s: %s seconds" %
                  (function.__name__ , str(t1 - t0)))
        return result

    return function_timer

class ImageAugment:
    def __init__(self, original_dataset_path, target_dataset_path):
        self.original_dataset_path = original_dataset_path
        self.target_dataset_path = target_dataset_path

    def create_augment_dataset(self, action, query_num):
        query_name_list = os.listdir(self.original_dataset_path)[0:query_num]
        for name in query_name_list:
            image_path = self.original_dataset_path + "/" + name
            image = cv2.imread(image_path)

            if action == "rotate":
                new_image = self._rotate_image(image)

            if action == "crop":
                new_image = self._crop_image(image)

            new_name = self.target_dataset_path + '/' + name
            cv2.imwrite(new_name, new_image)

    def _crop_image(self, image):
        # pad_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
        pad_image = image[0:50,0:45]
        return pad_image



    def _rotate_image(self, image_mat):
        (h, w) = image_mat.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        rotated_image = cv2.warpAffine(image_mat, M, (nW, nH))
        return  rotated_image



@fn_timer
def query_one_cnn_retrieval(query_path, index_file_path, uniform_size=(224, 224, 3)):
    # API: get one query result
    results = query.query(query_image_path=query_path,
                          index_file=index_file_path,
                          uniform_size=uniform_size,
                          limit=5)
    return results


@fn_timer
def retrieval_statistics(gallery_index_path, query_dataset_path,  uniform_size=(224, 224, 3)):
    accuracy = {}

    accuracy['first_right'] = 0
    accuracy['other_right'] = 0
    accuracy['wrong'] = 0

    query_name_list = os.listdir(query_dataset_path)

    for query_name in query_name_list:
        query_path = query_dataset_path + "/" + query_name
        results = query.query(query_path, gallery_index_path, uniform_size, limit=5)

        if results[0] == query_name:
            accuracy['first_right'] += 1
        elif query_name in results:
            accuracy['other_right'] += 1
        else:
            accuracy['wrong'] += 1
    return accuracy


if __name__ == '__main__':

    gallery_database_path = '${your_gallery_datase_path}'
    query_path = '${your_query_image_path}'
    index_file_path = 'featureCNN.h5'
    uniform_size = (64, 64, 3)

    # API: get all index
    gallery_index_path = index.index_images(
        gallery_database_path=gallery_database_path,
        index_file=index_file_path,
        uniform_size=uniform_size)

    result = query_one_cnn_retrieval(
        query_path=query_path,
        index_file_path=index_file_path,
        uniform_size=uniform_size)

    print("one query result:", result)

    accuracy = retrieval_statistics(
        gallery_index_path=gallery_index_path,
        query_dataset_path=gallery_database_path,
        uniform_size=uniform_size)

    print("accuracy statistics:", accuracy)
