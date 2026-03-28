# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
file(GLOB_RECURSE atb_plugin_operations_SRCS "./*.cpp")
set(CUST_PKG_PATH "$ENV{ASCEND_HOME_PATH}/opp/vendors/customize/op_api")
message(STATUS "atb_plugin_operations_SRCS: ${atb_plugin_operations_SRCS}")
include_directories(
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include
    ${CUST_PKG_PATH}/include  # 添加自定义算子头文件路径
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
link_directories(
    $ENV{ATB_HOME_PATH}/lib
    $ENV{ASCEND_HOME_PATH}/lib64
    ${CUST_PKG_PATH}/lib # 添加自定义算子so路径
    )
add_library(atb_plugin_operations STATIC ${atb_plugin_operations_SRCS})
target_link_libraries(atb_plugin_operations PRIVATE ${ACL_SHARED_LIBS} ${ACL_SHARED_LIBS} ascendcl)