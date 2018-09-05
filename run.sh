#! /bin/bash

#DEBUG="gdb --args "
DEBUG="cuda-gdb --args "
#DEBUG="ddd --debugger cuda-gdb --args "
#DEBUG="cgdb -d cuda-gdb --args "
#DEBUG="cuda-memcheck "

MODEL="./data/model/yolov3-voc-relu.caffemodel"
DEPLOY="./data/model/yolov3-voc-relu.prototxt"
# CALIBRATION="./data/model/CalibrationTable"
SYNSET="./data/model/voc.names"
IMAGELIST="./data/images/test.txt"

DEV_ID=$1
NMS=0.45       # $2
CONF=0.001     # $3
MODE=0         # 0 fp32, 1 fp16, 2 int8
BATCH_SIZE=1
N_ITERS=1

# Add this argument for INT8 inference.
# Note that only pascal GPU support INT8, like NVIDIA Tesla P4, P40


if [ ${MODE} -eq 2 ]
then
${DEBUG} ./bin/runYOLOv3 		-devID=${DEV_ID}			\
					-batchSize=${BATCH_SIZE}								\
					-nIters=${N_ITERS}											\
					-deployFile=${DEPLOY}										\
					-modelFile=${MODEL}											\
					-synsetFile=${SYNSET}										\
          -cali=${CALIBRATION}       						  \
          -imageFile=${IMAGELIST} 								\
          -nmsThreshold=${NMS}                    \
          -confThreshold=${CONF}
					#2>&1 | tee ./log/log.txt
else
${DEBUG} ./bin/runYOLOv3 		-devID=${DEV_ID}			\
					-batchSize=${BATCH_SIZE}								\
					-nIters=${N_ITERS}											\
					-deployFile=${DEPLOY}										\
					-modelFile=${MODEL}											\
					-synsetFile=${SYNSET}										\
          -imageFile=${IMAGELIST} 								\
          -nmsThreshold=${NMS}                    \
          -confThreshold=${CONF}
					#2>&1 | tee ./log/log.txt
fi
