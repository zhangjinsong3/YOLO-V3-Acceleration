# YOLO-V3-Acceleration
Using TensorRT to implement and accelerate YOLO v3. Multi-scale and NMS are included.  The acceleration ratio reaches 3 compared to the original darknet.
Model:
/data/model

Image:
/data/images

Build the sample:
$ make -j

Run the sample
$ ./run.sh


Plugin
===========================================

1. Upsample layer with nearest-neighbour interpolution. (Interp85 Interp97)


Bounding box parser
===========================================

* solution 1(used): launch reorgOutputKernel to fuse 3 output layers into 1 out layer form, but cost copy time, then do parser and NMS.

* solution 2(to be implement): iterate every output layer to do parser, then collect all bboxes to do NMS, also cost copy time during collection.

* solution 3(to be implement): create temp GPU memory to maintain a (float**) variable referring to 3 output layers, then do parser and NMS like ONE layer, based on index relation, but FAKE-ONE layer need to launch kernel 3 times.

