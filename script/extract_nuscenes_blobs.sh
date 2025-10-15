#!/usr/bin/env bash

# 将 nuScenes 官方拆分包顺序解压到指定目录。
# 默认假定压缩包位于 /mnt/new_data/nus，可按需使用环境变量覆盖。

set -euo pipefail

: "${SRC_DIR:=/mnt/new_data/nus}"
: "${DEST_DIR:=/Data/xvboxun/xbx/datasets/nuScence/}"

mkdir -p "${DEST_DIR}"

for idx in $(seq -w 1 10); do
  archive="${SRC_DIR}/v1.0-trainval${idx}_blobs.tgz"
  if [[ ! -f "${archive}" ]]; then
    echo "未找到压缩包: ${archive}" >&2
    exit 1
  fi
  echo "解压 ${archive} -> ${DEST_DIR}"
  tar zxvf "${archive}" -C "${DEST_DIR}"
done

meta_archive="${SRC_DIR}/v1.0-trainval_meta.tgz"
if [[ -f "${meta_archive}" ]]; then
  echo "解压 ${meta_archive} -> ${DEST_DIR}"
  tar zxvf "${meta_archive}" -C "${DEST_DIR}"
else
  echo "提示: 未找到 v1.0-trainval_meta.tgz，可忽略或稍后手动解压。" >&2
fi
