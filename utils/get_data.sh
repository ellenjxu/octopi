#!/bin/bash

TRAIN_DIR="data/train"
TEST_DIR="data/test"
BUCKET_URL_TRAIN="gs://malaria-training/init-train"
BUCKET_URL="gs://octopi-malaria-data/npy"
NEG_FILES="get_data_neg.txt"
DROPBOX_URL="https://dl.dropboxusercontent.com/scl/fo/hj6k3vof6j384ukw3xr3w/h?dl=1&e=1&st=t3xscwm5"

mkdir -p "$TRAIN_DIR/neg"
mkdir -p "$TRAIN_DIR/pos"
mkdir -p "$TEST_DIR/neg"
mkdir -p "$TEST_DIR/pos"

echo "Downloading train set..."

# old
# wget https://dl.dropboxusercontent.com/s/lfmu4mmashmc5z1bks8d0/parasite.npy?rlkey=kyg9nooylq0i614qvwdg6q1wy -O "$TRAIN_DIR/positive.npy"
# cleaned
gsutil cp gs://malaria-training/init-train/negative.npy "$TRAIN_DIR/neg/negative.npy"
gsutil cp gs://malaria-training/init-train/parasite_cleaned.npy "$TRAIN_DIR/pos/parasite_cleaned.npy"

echo "Downloading test set..."

while read -r filename; do
  gsutil cp "${BUCKET_URL}/${filename%.csv}.npy" "${TEST_DIR}/neg/"
done < "$NEG_FILES"

echo "Neg test samples downloaded"

wget "$DROPBOX_URL" -O "$TEST_DIR/pos.zip"
unzip "$TEST_DIR/pos.zip" -d "$TEST_DIR/pos"
rm "$TEST_DIR/pos.zip"

echo "Pos test samples downloaded"
