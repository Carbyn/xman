from PIL import Image
import os
import sys

def load_labels(superclass, date):
    labelspath = '../zsl_a_%s_train_%s/zsl_a_%s_train_annotations_labels_%s.txt' % (superclass.lower(), date, superclass.lower(), date)
    labels = {}
    for line in open(labelspath, 'r').readlines():
        items = list(map(lambda x: x.strip(), line.strip().split(',')))
        labels[items[-1]] = list(map(lambda x: int(x), (items[3], items[2][1:], items[5][:-1], items[4])))

    return labels

def main():
    if len(sys.argv) == 2:
        superclass = str(sys.argv[1])
    else:
        print("Param error")
        exit()

    date = '20180321'

    # ../zsl_a_animals_train_20180321/zsl_a_animals_train_images_20180321/A_ant
    imgdir_train = '../zsl_a_%s_train_%s/zsl_a_%s_train_images_%s' % (superclass.lower(), date, superclass.lower(), date)
    imgdir_train_crop = '../zsl_a_%s_train_%s_crop/zsl_a_%s_train_images_%s' % (superclass.lower(), date, superclass.lower(), date)

    labels = load_labels(superclass, date)

    categories = os.listdir(imgdir_train)
    for eachclass in categories:
        classpath = imgdir_train + '/' + eachclass
        if eachclass[0] == '.':
            continue
        for eachimg in os.listdir(classpath):
            if eachimg[0] == '.':
                continue

            classpath_crop = imgdir_train_crop + '/' + eachclass
            if not os.path.exists(classpath_crop):
                os.makedirs(classpath_crop, exist_ok=True)

            imgpath = classpath + '/' + eachimg
            imgpath_crop = classpath_crop + '/' + eachimg
            image = Image.open(imgpath)
            width = image.size[0]
            height = image.size[1]
            bbox = labels[eachclass + '/' + eachimg]
            image.crop(bbox).save(imgpath_crop)

if __name__ == "__main__":
    main()
