#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set -e

if [ -z "$BASEPATH" ]; then
  BASEPATH=$(cd "$(dirname $0)"; pwd)
fi
if [ -z "$THIRD_PARTY_CMAKE_DIR" ]; then
  THIRD_PARTY_CMAKE_DIR="cmake/third_party"
fi
OUTPUT_PATH="${BASEPATH}/third_party"
BUILD_RELATIVE_PATH="build/third_party"
BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}/"
if [ -z "$D_PKG_SERVER" ]; then
  THIRD_PARTY_PKG_PATH=""
else
  THIRD_PARTY_PKG_PATH="$(cd $BASEPATH && realpath $D_PKG_SERVER)"
fi
declare -A TARGET_ALIAS=(
    ["json"]="json_build"
)
# print usage message
usage() {
  echo "Usage:"
  echo "  sh build_third_party.sh [-h | --help] [-v | --verbose] [-j<N>]"
  echo "                          [--third_party_pkg_path=<PATH>]"
  echo "                          [--enable_github]"
  echo "                          [--output_path=<PATH>]"
  echo ""
  echo "Options:"
  echo "    -h, --help     Print usage"
  echo "    -v, --verbose  Display build command"
  echo "    -j<N>          Set the number of threads used for building third_party package, default is 8"
  echo "    --third_party_pkg_path=<PATH>"
  echo "                   Set third_party package path, default is empty"
  echo "    --enable_github"
  echo "                   Get third_party package from github, otherwise, from gitee"
  echo "    --output_path=<PATH>"
  echo "                   Set output path, default ./third_party"
  echo "    --targets=<LIST>"
  echo "                   Comma-separated list of build targets"
  echo "                           Available targets:"
  for alias in "${!TARGET_ALIAS[@]}"; do
    echo "                           $alias"
  done
  echo ""
}

# parse and set options
checkopts() {
  VERBOSE=""
  THREAD_NUM=8
  ENABLE_GITHUB="off"
  TARGETS=""

  # Process the options
  parsed_args=$(getopt -a -o j:hv -l help,verbose,third_party_pkg_path:,enable_github,output_path:,targets: -- "$@") || {
    usage
    exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      -j)
        THREAD_NUM="$2"
        shift 2
        ;;
      -v | --verbose)
        VERBOSE="VERBOSE=1"
        shift
        ;;
      --third_party_pkg_path)
        THIRD_PARTY_PKG_PATH="$(realpath $2)"
        shift 2
        ;;
      --enable_github)
        ENABLE_GITHUB="on"
        shift
        ;;
      --output_path)
        OUTPUT_PATH="$(realpath $2)"
        shift 2
        ;;
      --targets)
        IFS=',' read -ra raw_targets <<< "$2"
        formatted_targets=()
        invalid_targets=()
        for t in "${raw_targets[@]}"; do
            if [[ -z "${TARGET_ALIAS[$t]+_}" ]]; then
                invalid_targets+=("$t")
            else
                formatted_targets+=("${TARGET_ALIAS[$t]}")
            fi
        done
        if [ ${#invalid_targets[@]} -gt 0 ]; then
            echo -e "\033[31mError: Invalid target(s): ${invalid_targets[*]}\033[0m"
            usage
            exit 1
        fi
        TARGETS="${formatted_targets[*]}"
        TARGETS="${TARGETS// /;}"
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        echo "Undefined option: $1"
        usage
        exit 1
        ;;
    esac
  done
}

mk_dir() {
  local create_dir="$1"  # the target to make
  mkdir -pv "${create_dir}"
  echo "created ${create_dir}"
}

# build start
cmake_generate_make() {
  local build_path="$1"
  local cmake_args="$2"
  mk_dir "${build_path}"
  cd "${build_path}"
  echo "${cmake_args}"
  cmake ${cmake_args} "../../$THIRD_PARTY_CMAKE_DIR"
  if [ 0 -ne $? ]; then
    echo "execute command: cmake ${cmake_args} .. failed."
    exit 1
  fi
}

# create build path
build_third_party() {
  echo "create build directory and build third_party package"
  cd "${BASEPATH}"

  CMAKE_ARGS="-D D_PKG_SERVER=${THIRD_PARTY_PKG_PATH} \
              -D ENABLE_GITHUB=${ENABLE_GITHUB} \
              -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
              -D BUILD_TARGETS=${TARGETS}"

  echo "CMAKE_ARGS is: $CMAKE_ARGS"
  # make clean
  [ -d "${BUILD_PATH}" ] && rm -rf "${BUILD_PATH}"
  cmake_generate_make "${BUILD_PATH}" "${CMAKE_ARGS}"
  make ${VERBOSE} select_targets -j${THREAD_NUM} && make install
}

main() {
  cd "${BASEPATH}"
  checkopts "$@"
  mk_dir ${OUTPUT_PATH}
  build_third_party || { echo "third_party package build failed."; exit 1; }
  echo "---------------- third_party package build finished ----------------"
}

main "$@"
