import cPickle
import numpy as np
import cv2
import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

from scipy.ndimage.interpolation import zoom
import caffe
#import lmdb
import h5py
import random

caffe_root = '/home/weishen/dev/caffe/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')


import caffe

TRANSLATION_RANGE = 100
def rndTranslationAug(orgbox):
    rnd_trans = random.uniform(-100, 100)
    trans_box = []
    trans_box.append(orgbox[0] + rnd_trans)
    trans_box.append(orgbox[1] + rnd_trans)
    trans_box.append(orgbox[2] + rnd_trans)
    trans_box.append(orgbox[3] + rnd_trans)
    return trans_box

def faceDection():
    rect_width = 400
    face_box = [390, 110, 790, 510]#fixed for simplicity
    for video_idx in xrange(2, 5):

        video_path = '/home/weishen/dev/workspace/other/face-func/videos/'+str(video_idx) +'.mp4'
        capture = cv2.VideoCapture(video_path)

        cnt = 0
        face_folder = '/home/weishen/dev/workspace/other/face-func/face/video'+str(video_idx)+'/'
        while True:
            flag, frame = capture.read() # **EDIT:** to get frame size
            if flag==False:
                break
            roi = frame[face_box[1]:face_box[3], face_box[0]:face_box[2]]
            cv2.imwrite( face_folder + str(cnt) + '.jpg', roi)
            cnt+=1
            for iter_idx in  xrange(0, 10):
                trans_box = rndTranslationAug(face_box)
                roi = frame[trans_box[1]:trans_box[3], trans_box[0]:trans_box[2]]
                cv2.imwrite( face_folder + str(cnt) + '_' + str(iter_idx) + '.jpg', roi)

            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def FetchAllFiles(root_dir, suffix):
    file_names = []
    for root, dirs, files in os.walk(root_dir, True):
        for name in files:
            split_name = os.path.splitext(name)
            if split_name[1] == suffix:
                file_names.append(os.path.join(root, name))

    return file_names


def createDataset(video_idx):
    imgs_folder = '/home/weishen/dev/workspace/other/face-func/face/video'+str(video_idx)+'/'
    imgpath_list = FetchAllFiles(imgs_folder, '.jpg')
    label_interval = 6.0 / float(len(imgpath_list)) * 11

    #imgs_folder = '/home/weishen/dev/workspace/other/face-func/face/video'+str(2+video_idx)+'/'
    #imgpath_list.extend(FetchAllFiles(imgs_folder, '.jpg'))

    img_list = []
    label_list = []
    output_db_path = '/home/weishen/dev/workspace/other/face-func/facedb'+str(video_idx)+'_64.h5'
    output_db_p_path = '/home/weishen/dev/workspace/other/face-func/facedb'+str(video_idx)+'_64_p.h5'
    for imgpath in imgpath_list:
        rawimg = mpimg.imread(imgpath)
        #np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))
        img = []
        img.append(zoom(rawimg[:,:,0], 64.0/400.0))#
        img.append(zoom(rawimg[:,:,1], 64.0/400.0))
        img.append(zoom(rawimg[:,:,2], 64.0/400.0))
        img = np.asarray(img).astype(np.float)/255
        #print img.shape
        #imgplot = plt.imshow(img[0])
        #plt.show()
        img_name = os.path.splitext(os.path.split(imgpath)[1])[0]
        img_idx = int(img_name.split('_')[0])
        label = img_idx*label_interval
        img_list.append(img)
        label_list.append(label)

        #imgplot = plt.imshow(img[0])
        #plt.show()

    f = h5py.File(output_db_path, 'w')
    f['data'] = img_list
    f['label'] = label_list
    f.close()

    f = h5py.File(output_db_p_path, 'w')
    f['data_p'] = img_list
    f['label_p'] = label_list
    f.close()

    print 'Done'

def testDB():
    dbpath = '/home/weishen/dev/workspace/other/face-func/face/facedb.h5'
    f = h5py.File(dbpath, 'r')
    img_list = f['data']
    label_list = f['label']
    print len(img_list)
    print img_list[0].shape
    print len(label_list)
    print label_list
    f.close()

def testModel():
    caffe.set_mode_cpu()
    net = caffe.Net('/home/weishen/dev/workspace/other/face-func/caffe-model/face_siamese.prototxt',
                '/home/weishen/dev/workspace/other/face-func/caffe-model/face_no_siamese_iter_50000.caffemodel',
                caffe.TEST)
    dbpath = '/home/weishen/dev/workspace/other/face-func/facedb3_64.h5'
    f = h5py.File(dbpath, 'r')
    img_list = f['data']
    label_list = f['label']

    net.blobs['data'].reshape(len(img_list),3,64,64)
    net.blobs['data'].data[...] = img_list
    out = net.forward()['feat']

    dif_list = []
    for img_idx in xrange(0, out.shape[0]):
        dif_list.append(abs(out[img_idx][0] - label_list[img_idx]))
        #print str(out[img_idx]) + ' vs ' + str(label_list[img_idx])
    print np.histogram(dif_list, bins=np.arange(0, 6, 0.1))
    print 'Done'

if __name__ == '__main__':
    #faceDection()
    #createDataset(3)
    #testDB()
    testModel()