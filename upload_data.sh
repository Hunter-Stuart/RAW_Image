#!/bin/bash
# ============================================================
# AODRaw 训练数据：打包 → 切分 → 上传 → 校验 → 服务器合并解压
# 新服务器版 (mygpu2)
# 前置: 1) ~/.ssh/config 已配置 mygpu2 别名（空口令密钥 id_mygpu）
#       2) 在 /d/firstwork/RAW_Image 目录下，AODRaw_downsampled_srgb 完整(7785张JPG)
# 用法: bash upload_data.sh
# ============================================================
set -u

HOST="mygpu2"
REMOTE_DIR="/home/zeng"
TAR="AODRaw_images.tar.gz"
PARTS="part_"

cd "$(dirname "$0")"

echo "==== [0] 检查图片数量（应为 7785） ===="
N=$(find AODRaw_downsampled_srgb -name "*.JPG" | wc -l)
echo "本地 JPG 数量: $N"
if [ "$N" -ne 7785 ]; then
  echo "!!! 数量不对，请检查图片目录后重跑"
  exit 1
fi

echo "==== [1] 打包 tar.gz ===="
rm -f "$TAR"
tar -czf "$TAR" AODRaw_downsampled_srgb
ls -lh "$TAR"
TL=$(tar -tzf "$TAR" | wc -l)
echo "包内条目数: $TL（应约 7788）"

echo "==== [2] 切分 200MB/卷 ===="
rm -f ${PARTS}*
split -b 200M "$TAR" "$PARTS"
NP=$(ls ${PARTS}* | wc -l)
echo "分卷数: $NP（预期 22）"

echo "==== [3] 上传分卷（每卷最多重试8次） ===="
ssh -o ConnectTimeout=15 $HOST "mkdir -p $REMOTE_DIR/parts" || {
  echo "!!! SSH 连接失败（检查网络/密钥），退出"
  exit 1
}
for f in ${PARTS}*; do
  ok=0
  for i in 1 2 3 4 5 6 7 8; do
    scp -o ConnectTimeout=15 -o ServerAliveInterval=10 \
        "$f" "$HOST:$REMOTE_DIR/parts/" && { ok=1; break; }
    echo "    [$f] 第${i}次失败，3秒后重试"
    sleep 3
  done
  if [ "$ok" -eq 1 ]; then
    echo "    [$f] 完成"
  else
    echo "!!! [$f] 8次重试均失败 —— 直接重新运行本脚本即可"
  fi
done

echo "==== [4] MD5 完整性校验 ===="
# 注意: Windows Git Bash 与 Linux 的 md5sum 输出格式有差异(*标记/换行)，
# 统一加 -b 强制二进制格式，并只对比 哈希+文件名 两列
md5sum -b ${PARTS}* | sort -k2 | awk '{print $1, $NF}' > local_by_name.md5
ssh -o ConnectTimeout=15 $HOST \
    "cd $REMOTE_DIR/parts && md5sum -b ${PARTS}*" | sort -k2 | awk '{print $1, $NF}' > remote_by_name.md5
if diff -q local_by_name.md5 remote_by_name.md5 > /dev/null 2>&1; then
  echo "校验通过：服务器分卷与本地完全一致"
else
  echo "!!! 校验不一致，执行 diff local_by_name.md5 remote_by_name.md5 找出差异分卷后重跑本脚本"
  exit 1
fi

echo "==== [5] 服务器合并 + 解压 + 计数 ===="
ssh -o ConnectTimeout=15 $HOST \
    "cd $REMOTE_DIR/parts && cat ${PARTS}* > $REMOTE_DIR/$TAR && tar -tzf $REMOTE_DIR/$TAR | wc -l"
ssh -o ConnectTimeout=15 $HOST \
    "tar -xzf $REMOTE_DIR/$TAR -C $REMOTE_DIR/ && find $REMOTE_DIR/AODRaw_downsampled_srgb -name '*.JPG' | wc -l"

echo ""
echo "==== 完成 ===="
echo "确认上面最后两个数字分别是: 包内条目≈7788、解压JPG数=7785"
echo "确认无误后清理临时文件:"
echo "  ssh $HOST 'rm -rf $REMOTE_DIR/parts $REMOTE_DIR/$TAR'"
echo "  rm -f ${PARTS}* $TAR local_by_name.md5 remote_by_name.md5"
