# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg as LA

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

class VGGNet:
    def __init__(self, input_shape=(224, 224, 3)):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = input_shape
        self.weight = 'imagenet' # download weights
        self.pooling = 'max' # include_top=True, output=(1, 1000); False, (1, 521)
        self.model = VGG16(weights=self.weight,
                           input_shape=(self.input_shape[0],
                                          self.input_shape[1],
                                          self.input_shape[2]),
                           pooling=self.pooling,
                           include_top=False)
        self.model.predict(np.zeros((1,) + input_shape))

    def extract_feat(self, img_path):
        '''
        Use vgg16 model to extract features
        Output normalized feature vector
        TODO: try vgg19
        :param img_path:
        :return:
        '''
        img = image.load_img(img_path, target_size=(self.input_shape[0],
                                                    self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img) # feat.shape = (1, 512)
        norm_feat = feat[0]/LA.norm(feat[0]) # L2 norm
        return norm_feat