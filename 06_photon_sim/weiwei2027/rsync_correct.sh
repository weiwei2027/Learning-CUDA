#!/bin/bash
# 修正后的 rsync 命令 - 不排除源代码！

# 问题：--exclude='photon_sim_*' 会排除 src/photon_sim_*.cu 源文件
# 解决：只排除根目录的可执行文件，不排除 src/ 目录下的源代码

cd 06_photon_sim/weiwei2027

rsync -avz --progress \
    -e "ssh -p $REMOTE_PORT -i $KEY_PATH" \
    --exclude='build/' \
    --exclude='cmake-build-*/' \
    --exclude='output/*.bin' \
    --exclude='output/*.log' \
    --exclude='photon_sim_nv' \
    --exclude='photon_sim_cpu' \
    --exclude='photon_sim_iluvatar' \
    --exclude='photon_sim_metax' \
    --exclude='photon_sim_moore' \
    --exclude='*.o' \
    --exclude='*.exe' \
    --exclude='.git/' \
    --exclude='.idea/' \
    --exclude='remote_results_*/' \
    ./ \
    $REMOTE_HOST:/home/weiwei/weiwei2027/
