cmake_minimum_required(VERSION 3.16)
project("test_model")
option(USE_CXX11_ABI "USE_CXX11_ABI" OFF)
set(CMAKE_BUILD_TYPE Debug)
if(USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(CUST_PKG_PATH "$ENV{ASCEND_HOME_PATH}/opp/vendors/customize/op_api")

include_directories(
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include
    ${CUST_PKG_PATH}/include  # 添加自定义算子头文件路径
    ${CMAKE_CURRENT_SOURCE_DIR}
    ./src/
    )

message(".h path =  ${CUST_PKG_PATH}/include")
link_directories(
    $ENV{ATB_HOME_PATH}/lib
    $ENV{ASCEND_HOME_PATH}/lib64
    ${CUST_PKG_PATH}/lib # 添加自定义算子so路径
    )


link_libraries(atb_plugin_operations)
add_executable(test_model ./src/main.cpp)

# 列出所有的头文件目录
target_link_libraries(test_model PRIVATE atb ascendcl cust_opapi opapi nnopbase pthread) # 添加自定义算子so及适配so





