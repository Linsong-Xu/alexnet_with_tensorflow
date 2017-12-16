import scipy.misc
import cv2
import numpy as np
import os

datas = []
images_path = './images'

with open('train.csv') as file:
    for line_id, line in enumerate(file):
        label, feat = line.split(',')
        feat = np.fromstring(feat, dtype=int, sep=' ')
        feat = np.reshape(feat, (48,48,1))
        datas.append((feat, int(label), line_id))
# 28709 images
feats, labels, line_ids = zip(*datas)
feats = np.asarray(feats)

if not os.path.isdir(images_path):
    os.makedirs(images_path)

for i in range(len(feats)):
    scipy.misc.imsave('./images/{}.jpg'.format(i), np.mat(feats[i]))
    img1 = cv2.imread('./images/{}.jpg'.format(i))
    res2 = cv2.resize(img1,(227, 227), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite('./images/{}.jpg'.format(i),res2)
    print(i)

# we use about 80% images as train data and others as valid data
with open('./images/train.txt', 'w') as f:
    for i in range(22960):
        f.write('./images/{}.jpg {}\n'.format(i,labels[i]))
        print(i)

with open('./images/valid.txt', 'w') as f:
    for i in range(22960, 28709):
        f.write('./images/{}.jpg {}\n'.format(i,labels[i]))
        print(i)

#mean = np.mean(feats)
#print(mean)
