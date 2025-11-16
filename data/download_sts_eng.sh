#!/bin/bash
cd data


wget https://dl.fbaipublicfiles.com/glue/data/STS-B.zip
unzip STS-B.zip
rm STS-B.zip

cd STS-B
rm dev.tsv test.tsv train.tsv LICENSE.txt readme.txt

cd /workspace/simcse/SimCSE-multilanguages/

# bash data/download_sts_eng.sh


# data/sts_B/original/sts-dev.tsv
# data/sts_B/original/sts-test.tsv
# data/sts_B/original/sts-train.tsv