#!/bin/bash
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
BASE_DIR=$(pwd)

# 编译wheel包
if ! python3 setup.py build bdist_wheel; then
    echo "Error: Failed to build wheel package"
    exit 1
fi

# 使用文件锁保护pip安装过程
mkdir -p ~/tmp
LOCK_FILE="${HOME}/tmp/pip_install.lock"
(
    flock -x 200
    echo "Acquired lock for pip install"
    
    # 安装wheel包
    cd ${BASE_DIR}/dist
    pip3 install *.whl --force-reinstall
    
    echo "Released lock for pip install"
) 200>"$LOCK_FILE"

