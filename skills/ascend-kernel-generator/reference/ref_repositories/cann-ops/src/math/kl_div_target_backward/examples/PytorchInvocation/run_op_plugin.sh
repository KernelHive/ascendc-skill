#!/bin/bash
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

if [ ! $ASCEND_HOME_PATH ]; then
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
source $ASCEND_HOME_PATH/bin/setenv.bash

# 当前示例使用Python-3.9版本
PYTHON_VERSION=$(python3 -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1"."$2}')
if [ "$PYTHON_VERSION" != "3.9" ]; then
    echo "Error: Python3 version is not 3.9"
    exit 1
fi
# 当前示例使用Pytorch-2.1.0版本
PYTORCH_VESION=$(pip3 show torch | grep "Version:" | awk '{print $2}' | awk -F '.' '{print $1"."$2"."$3}' | awk -F '+' '{print $1}')
if [ "$PYTORCH_VESION" != "2.1.0" ]; then
    echo "Error: Pytorch version is not 2.1.0"
    exit 1
fi
export HI_PYTHON=python${PYTHON_VERSION}
export PYTHONPATH=$ASCEND_HOME_PATH/python/site-packages:$PYTHONPATH
export PATH=$ASCEND_HOME_PATH/python/site-packages/bin:$PATH

function main() {
    # 1. 清除遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*

    # 2. 下载PTA源码仓，必须要git下载
    cd ${CURRENT_DIR}
    PYTORCH_DIR="${CURRENT_DIR}/pytorch"
    rm -rf $PYTORCH_DIR
    git clone https://gitee.com/ascend/pytorch.git

    cd ${PYTORCH_DIR}
    git checkout -b v2.1.0 origin/v2.1.0
    git submodule update --init --recursive
    # 不编译Tensorpipe子仓，会存在编译依赖
    rm -rf "third_party/Tensorpipe/"

    cd ${CURRENT_DIR}
    # 3. PTA自定义算子注册
    FUNCTION_REGISTE_FIELD="op_plugin_patch/op_plugin_functions.yaml"
    FUNCTION_REGISTE_FILE="${PYTORCH_DIR}/third_party/op-plugin/op_plugin/config/op_plugin_functions.yaml"
    line="  - func: npu_kl_div_target_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, bool log_target) -> Tensor"
    if ! grep -q "$line" $FUNCTION_REGISTE_FILE; then
        sed -i "/custom:/r   $FUNCTION_REGISTE_FIELD" $FUNCTION_REGISTE_FILE
    fi

    # 4. 编译PTA插件并安装
    cp -rf op_plugin_patch/*.cpp ${PYTORCH_DIR}/third_party/op-plugin/op_plugin/ops/opapi
    export DISABLE_INSTALL_TORCHAIR=FALSE
    cd ${PYTORCH_DIR}
    (bash ci/build.sh --python=${PYTHON_VERSION} --disable_rpc ; pip uninstall torch-npu -y ; pip3 install dist/*.whl)

    # 5. 执行测试文件
    cd ${CURRENT_DIR}
    export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
    python3 test_ops_custom.py
    if [ $? -ne 0 ]; then
        echo "ERROR: run KlDivTargetBackward op failed!"
        return 1
    fi
    echo "INFO: Ascend C KlDivTargetBackward SUCCESS"
}
main
