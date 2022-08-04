#!/bin/bash

HCPDIR="/data/hcp-plis/hdd01/"
TARGETDIR="/data/users2/vjoshi6/aibm_labels/"
CSVFILE=$1


find $HCPDIR -name '*aparc.stats' > /data/users2/vjoshi6/bin/pythonFiles/AIBM/training/data/aibm.stats.csv
while read line
do
    A=$(echo $line | awk -F/ '{print $5}')
    echo "Copying: ${line} to ${TARGETDIR}$A"
    mkdir -p ${TARGETDIR}$A
    cp $line ${TARGETDIR}$A
done < $CSVFILE

