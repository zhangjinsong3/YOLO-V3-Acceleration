#! /bin/bash

result=mAP.csv
mAP=0.0
for nms in `seq 0.4 0.05 0.7`
do
    for conf in `seq 0.001 0.001 0.005`
    do
        ./run.sh 0 ${nms} ${conf}
        cd results/
        mAP=`python calc_mAP.py 0.5  2>&1 1>/dev/null`
        echo $( printf '%f %f %f' ${nms} ${conf} ${mAP} ) >> ${result}
        cat ${result}
        rm -rf cache comp4_det_test_*
        cd ../
    done
done
