#!/bin/bash

# This script downloads the HAI dataset.

wget https://raw.githubusercontent.com/icsdataset/hai/master/hai-20.07/train2.csv.gz
wget https://raw.githubusercontent.com/icsdataset/hai/master/hai-20.07/test2.csv.gz

gzip -d train2.csv.gz
gzip -d test2.csv.gz