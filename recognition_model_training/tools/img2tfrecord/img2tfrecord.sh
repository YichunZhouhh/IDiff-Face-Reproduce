#!/bin/bash

if ! pip show tensorflow &> /dev/null; then
  pip install tensorflow
fi

pip install --upgrade protobuf==3.20.1

# DATA_PATH="/remote-home/share/yxmi/datasets/"
DATA_PATH="/remote-home/yczhou/CVPR24_FRCSyn_ADMIS/dataset"
SRC_NAME=""
PATH_SUFFIX="images"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --data_path=*)
      DATA_PATH="${1#*=}"
      shift
      ;;
    --src_name=*)
      SRC_NAME="${1#*=}"
      shift
      ;;
    *)
      echo "unexpected parameter: $1"
      exit 1
      ;;
  esac
done

if [ -z "$SRC_NAME" ]; then
  echo "Source image directory not specified"
  exit 1
else
  echo "Source image directory: ${DATA_PATH}${SRC_NAME}"
fi

python create_img_list.py --data_path=${DATA_PATH} --src_name=${SRC_NAME} --path_suffix=${PATH_SUFFIX}

python img2tfrecord.py --img_list="${DATA_PATH}${SRC_NAME}/${SRC_NAME}_img_list.txt" --src_img_dir="${DATA_PATH}${SRC_NAME}" --output_dir="${DATA_PATH}" --tfrecords_name="TFR-${SRC_NAME}"