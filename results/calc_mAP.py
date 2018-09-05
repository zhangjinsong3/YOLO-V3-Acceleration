#!/usr/bin/env python
import sys
from voc_eval import voc_eval


names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
         'bus', 'car', 'cat', 'chair', 'cow',
         'diningtable', 'dog', 'horse', 'motorbike', 'person',
         'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

iou_threshold = float(sys.argv[1])
print 'IOU threshold %.5f' % iou_threshold

mAP = []
for name in names:
    recall, precision, ap =  voc_eval(
        # change this to your results file
        './comp4_det_test_{}.txt',
        # change these 2 to your voc dataset
        '/home/weisong/_yolov3/YOLOv3-darknet/data_voc/VOCdevkit/VOC2007/Annotations/{}.xml',
        '/home/weisong/_yolov3/YOLOv3-darknet/data_voc/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        name,
        './cache/',
        iou_threshold)

    print "%-15s %.5f" % (name, ap)
    mAP.append(ap)

ret = (float)(sum(mAP) / len(mAP))
print 'mAP = %.5f' % ret
exit(ret)
