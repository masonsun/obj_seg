import os
import sys
import numpy as np
from PIL import Image


class DavisDataset:
    def __init__(self, train_list, test_list, root='.', data_aug=False):
        """Load DAVIS 2017 dataset object, based on code from @scaelles

        :param train_list: textfile or list with paths to images for training
        :param test_list: textfile or list with paths to images for testing
        :param root: root of path to images and labels
        :param data_aug: option to include data augmentation
        """

        # load images and labels
        print("Loading files...")

        # training
        if not isinstance(train_list, list) and train_list is not None:
            with open(train_list, 'r+') as f:
                train_paths = f.readlines()
        elif isinstance(train_list, list):
            train_paths = train_list
        else:
            train_paths = []

        # testing
        if not isinstance(test_list, list) and test_list is not None:
            with open(test_list, 'r+') as f:
                test_paths = f.readlines()
        elif isinstance(test_list, list):
            test_paths = test_list
        else:
            test_paths = []

        # data containers
        self.images_train, self.images_train_path = [], []
        self.labels_train, self.labels_train_path = [], []
        self.images_test, self.images_test_path = [], []

        ############
        # TRAINING #
        ############
        for index, line in enumerate(train_paths):
            img = Image.open(os.path.join(root, str(line.split()[0])))
            lab = Image.open(os.path.join(root, str(line.split()[1])))
            img.load()
            lab.load()
            lab = lab.split()[0]

            if data_aug:
                # types of data augmentation
                aug_scale = [0.5, 0.8, 1.0]
                aug_flip = True

                if index == 0:
                    sys.stdout.write('Augmenting data...')
                for s in aug_scale:
                    img_size = tuple([int(img.size[0] * s), int(img.size[1] * s)])
                    img_scale, lab_scale = img.resize(img_size), lab.resize(img_size)
                    self.images_train.append(np.array(img_scale, dtype=np.uint8))
                    self.labels_train.append(np.array(lab_scale, dtype=np.uint8))

                    if aug_flip:
                        img_scale_flip = img_scale.transpose(Image.FLIP_LEFT_RIGHT)
                        lab_scale_flip = lab_scale.transpose(Image.FLIP_LEFT_RIGHT)
                        self.images_train.append(np.array(img_scale_flip, dtype=np.uint8))
                        self.labels_train.append(np.array(lab_scale_flip, dtype=np.uint8))
            # no data augmentation
            else:
                if index == 0:
                    sys.stdout.write('Loading data...')
                self.images_train.append(np.array(img, dtype=np.uint8))
                self.labels_train.append(np.array(lab, dtype=np.uint8))

            # load paths to images and labels
            if (index + 1) % 20 == 0:
                sys.stdout.write('{} training images processed.'.format(index + 1))
            self.images_train_path.append(os.path.join(root, str(line.split()[0])))
            self.labels_train_path.append(os.path.join(root, str(line.split()[1])))

        # convert lists to numpy array representation
        self.images_train_path = np.array(self.images_train_path)
        self.labels_train_path = np.array(self.labels_train_path)

        ###########
        # TESTING #
        ###########
        for index, line in enumerate(test_paths):
            self.images_test.append(np.array(Image.open(os.path.join(root, str(line.split()[0]))), dtype=np.uint8))
            self.images_test_path.append(os.path.join(root, str(line.split()[0])))
            if (index + 1) % 100 == 0:
                sys.stdout.write('{} testing images processed'.format(index + 1))
        print('Finished processing dataset')

        # dataset parameters
        self.train_counter = 0
        self.test_counter = 0
        self.train_size = max(len(self.images_train_path), len(self.images_train))
        self.test_size = len(self.images_test_path)
        self.train_index = np.arange(self.train_size)
        np.random.shuffle(self.train_index)

    def next_batch(self, batch_size, phase='train'):
        """Get next batch of images and labels

        :param batch_size: size of batch
        :param phase: options 'train' or 'test'
        :return:
            training: tuple that contains list of images and labels, respectively, as numpy arrays
            testing: image as numpy array and list of image paths
        """

        if phase == 'train':
            if self.train_counter + batch_size < self.train_size:
                index = np.array(self.train_index[self.train_counter : self.train_counter + batch_size])
                imgs = [self.images_train[l] for l in index]
                labs = [self.labels_train[l] for l in index]
                self.train_counter += batch_size
            # handle leftover images
            else:
                prev_index = np.array(self.train_index[self.train_counter : ])
                np.random.shuffle(self.train_index)
                self.train_counter = (self.train_counter + batch_size) % self.train_size
                index = np.array(self.train_index[ : self.train_counter])
                imgs = [self.images_train[l] for l in prev_index] + [self.images_train[l] for l in index]
                labs = [self.labels_train[l] for l in prev_index] + [self.labels_train[l] for l in index]
            return imgs, labs
        # testing
        else:
            if self.test_counter + batch_size < self.test_size:
                imgs = self.images_test[self.test_counter : self.test_counter + batch_size]
                paths = self.images_test_path[self.test_counter : self.test_counter + batch_size]
                self.test_counter += batch_size
            # handle leftover images
            else:
                new_counter = (self.test_counter + batch_size) % self.test_size
                imgs = self.images_test[self.test_counter : ] + self.images_test[ : new_counter]
                paths = self.images_test_path[self.test_counter : ] + self.images_test_path[ : new_counter]
                self.test_counter = new_counter
            return imgs, paths

    def get_train_size(self):
        return self.train_size

    def get_test_size(self):
        return self.test_size

    def get_train_img_size(self):
        x = self.images_train[self.train_counter]
        if isinstance(x, np.ndarray):
            return x.shape
        else:
            w, h = Image.open(x).size
            return h, w