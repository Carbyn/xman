#!/usr/bin/env python
# coding=utf-8
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import DirectoryIterator, load_img, img_to_array
import numpy as np
from keras import backend as K
from utils import attrs_reduce 
import os

class XmanImageDataGenerator(ImageDataGenerator):
   def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode=None,
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        return XmanDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)

class XmanDirectoryIterator(DirectoryIterator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            print(fname)
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else :
            batch_y = []
            for i, j in enumerate(index_array):
                fname = self.filenames[j]
                attrs = self._get_attrs(fname)
                batch_y.append(attrs)
        batch_y = np.array(batch_y)
        return batch_x, batch_y

    def _attrstr2list(self, s):
        s = s[1:-2]
        tokens = s.split()
        attrlist = list()
        for each in tokens:
            attrlist.append(float(each))
        return attrlist
    def _get_attrs(self, fname):
        img_class_name = '_'.join(fname.split('/')[0].split('_')[1:])
        superclass_prefix = '_'.join(fname.split('/')[0][0])
        if not hasattr(self,'attrs_map'):
            superclass_map = {'A':'Animals', 'F':'Fruits'}
            superclass = superclass_map[superclass_prefix]
            self.attrs_map = {}
            date = '20180321'
            attributes_path = '../zsl_a_%s_train_%s/zsl_a_%s_train_annotations_attributes_per_class_%s.txt' % (superclass.lower(), date, superclass.lower(), date)
            labels_list_path = '../zsl_a_%s_train_%s/zsl_a_%s_train_annotations_label_list_%s.txt' % (superclass.lower(), date, superclass.lower(), date)
            entropy_thr = 0 

            #获取标签名称到类别的映射
            flabels = open(labels_list_path, 'r', encoding='UTF-8')
            lines_label = flabels.readlines()
            flabels.close()
            labels = dict()
            for each in lines_label:
                tokens = each.split(', ')
                class_type = tokens[0]
                class_name = tokens[1]
                labels[class_type] = class_name
            label_attrs = attrs_reduce(attributes_path, superclass, entropy_thr)
            for class_type, attrs  in label_attrs.items():
                if class_type not in labels.keys():
                    continue
                class_name = labels[class_type]
                self.attrs_map[class_name] = attrs
        return self.attrs_map[img_class_name]


        
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        superclass = sys.argv[1]
    else:
        print('Parameters error')
        exit()

    train_datagen = XmanImageDataGenerator(
        rescale=1./255,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=True)
    trainpath = 'trainval_'+superclass+'/train'
    train_generator = train_datagen.flow_from_directory(
        trainpath,
        target_size=(72, 72),
        batch_size=32)
    for x,y in train_generator:
        print(np.shape(x))
        print(x[0,0,0])
        print(y)
        print(np.shape(y))
        exit()

