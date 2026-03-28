#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=0

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

# 导出环境变量
DTYPE="float16"
JSON_NAME=add_custom
SHORT=m:,v:,
LONG=is-dynamic:,dtype:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :; do
    case "$1" in
    # IS_DYNAMIC 0: static op
    # IS_DYNAMIC 1: dynamic op
    -m | --is-dynamic)
        IS_DYNAMIC="$2"
        shift 2
        ;;
    # float16, float, int32
    -v | --dtype)
        DTYPE="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done
if [ ! $IS_DYNAMIC ]; then
    IS_DYNAMIC=1
fi

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/lib64

# 检查当前昇腾芯片的类型
function check_soc_version() {
    SOC_VERSION_CONCAT=$(python3 -c '''
import ctypes, os
def get_soc_version():
    max_len = 256
    rtsdll = ctypes.CDLL(f"libruntime.so")
    c_char_t = ctypes.create_string_buffer(b"\xff" * max_len, max_len)
    rtsdll.rtGetSocVersion.restype = ctypes.c_uint64
    rt_error = rtsdll.rtGetSocVersion(c_char_t, ctypes.c_uint32(max_len))
    if rt_error:
        print("rt_error:", rt_error)
        return ""
    soc_full_name = c_char_t.value.decode("utf-8")
    find_str = "Short_SoC_version="
    ascend_home_dir = os.environ.get("DDK_PATH")
    with open(f"{ascend_home_dir}/compiler/data/platform_config/{soc_full_name}.ini", "r") as f:
        for line in f:
            if find_str in line:
                start_index = line.find(find_str)
                result = line[start_index + len(find_str):].strip()
                return "{},{}".format(soc_full_name, result.lower())
    return ""
print(get_soc_version())
    ''')
    if [[ ${SOC_VERSION_CONCAT}"x" = "x" ]]; then
        echo "ERROR: SOC_VERSION_CONCAT is invalid!"
        return 1
    fi
    SOC_FULL_VERSION=$(echo $SOC_VERSION_CONCAT | cut -d ',' -f 1)
    SOC_SHORT_VERSION=$(echo $SOC_VERSION_CONCAT | cut -d ',' -f 2)
}

function main {
    if [[ ${IS_DYNAMIC}"x" = "x" ]]; then
        echo "ERROR: IS_DYNAMIC is invalid!"
        return 1
    fi

    # 1. 清楚遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*
    rm -rf op_models/*.om

    # 2. 编译离线om模型
    cd $CURRENT_DIR
    if [ $IS_DYNAMIC == 1 ]; then
        atc --singleop=scripts/${JSON_NAME}_dynamic_shape.json --output=op_models/ --soc_version=${SOC_FULL_VERSION}
    else
        atc --singleop=scripts/${JSON_NAME}_static_shape.json --output=op_models/ --soc_version=${SOC_FULL_VERSION}
    fi

    # 3. 生成输入数据和真值数据
    cd $CURRENT_DIR
    python3 scripts/gen_data.py
    if [ $? -ne 0 ]; then
        echo "ERROR: generate input data failed!"
        return 1
    fi
    echo "INFO: generate input data success!"

    # 4. 编译acl可执行文件
    cd $CURRENT_DIR
    rm -rf build
    mkdir -p build
    cd build
    cmake ../src
    if [ $? -ne 0 ]; then
        echo "ERROR: cmake failed!"
        return 1
    fi
    echo "INFO: cmake success!"
    make
    if [ $? -ne 0 ]; then
        echo "ERROR: make failed!"
        return 1
    fi
    echo "INFO: make success!"

    # 5. 运行可执行文件
    cd $CURRENT_DIR/output
    if [ $IS_DYNAMIC == 1 ]; then
        echo "INFO: execute dynamic op!"
        ./execute_add_op $IS_DYNAMIC 2048
    else
        echo "INFO: execute static op!"
        ./execute_add_op
    fi
    if [ $? -ne 0 ]; then
        echo "ERROR: acl executable run failed! please check your project!"
        return 1
    fi
    echo "INFO: acl executable run success!"

    # 6. 比较真值文件
    cd $CURRENT_DIR
    python3 scripts/verify_result.py output/output_z.bin output/golden.bin
    if [ $? -ne 0 ]; then
        echo "ERROR: verify result failed!"
        return 1
    fi
}

check_soc_version
main
