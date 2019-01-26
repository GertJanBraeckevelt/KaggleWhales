import cv2
import numpy as np
from glob import glob
import os


class Generator(object):
    """
        A generator to select 3 images (anchor, positive, negative) to be trained using triplet loss embedding
    """
    def __init__(self,
                 cls_paths,
                 target_dim = 224,
                 aug = None,
                 batch_size=32,
                 preprocess = lambda x: x):

        self.target_dim = target_dim
        self.batch_size = batch_size

        self.cls_paths = cls_paths
        self.num_cls = len(cls_paths)

        self.augmenter = aug
        self.preprocess = preprocess

    def next(self):

        batch_images = np.zeros(shape=(self.batch_size, self.target_dim, self.target_dim, 3),dtype=np.uint8)
        batch_annotations = np.zeros(shape=(self.batch_size,),dtype=np.float32)

        options = []
        picked_cls = None
        chosen_file = None

        while len(options) < 2:
            picked_cls = np.random.randint(self.num_cls)
            options = glob(os.path.join(self.cls_paths[picked_cls], '*'))
            chosen_file = np.random.choice(options)

        batch_images[0] = self.read(chosen_file)
        batch_annotations[0] = picked_cls

        if np.random.rand() > 0.2:
            options.remove(chosen_file)

        chosen_file = np.random.choice(options)
        batch_images[1] = self.read(chosen_file)
        batch_annotations[1] = picked_cls


        for b in range(2,self.batch_size):

            if np.random.rand() > 0.1 and len(options) > 0 and b < self.batch_size-1:

                chosen_file = np.random.choice(options)

            else:

                picked_cls = np.random.randint(self.num_cls)
                options = glob(os.path.join(self.cls_paths[picked_cls], '*'))
                chosen_file = np.random.choice(options)

            if np.random.rand() > 0.2:
                options.remove(chosen_file)

            batch_images[b] = self.read(chosen_file)
            batch_annotations[b] = picked_cls

        if self.augmenter:
            batch_images = self.augmenter.augment_images(batch_images)

        return [batch_images.astype(np.float)/255], [np.squeeze(batch_annotations)]

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self.next()

    def read(self,path):
        img = cv2.imread(path)[:,:,::-1]
        img = self.preprocess(img)
        img = cv2.resize(img, (self.target_dim, self.target_dim))

        return img

    def __len__(self):
        return len(self.cls_paths)
