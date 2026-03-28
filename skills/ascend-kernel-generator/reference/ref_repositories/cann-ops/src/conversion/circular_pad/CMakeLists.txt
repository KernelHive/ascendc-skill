add_ops_compile_options(
        OP_NAME CircularPad
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

#target_sources(op_host_aclnn PRIVATE
#op_host/circular_pad.cpp
#)

#optiling
target_sources(optiling PRIVATE
        op_host/circular_pad.cpp
        op_host/circular_pad_common.cc
)

target_include_directories(optiling PRIVATE
        $CMAKECURRENTSOURCEDIR/ophost
        ${CMAKE_SOURCE_DIR}/src/common/inc
        $ASCENDCANNPACKAGEPATH/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        $ASCENDCANNPACKAGEPATH/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        $ASCENDCANNPACKAGEPATH/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

#opproto
target_sources(opsproto PRIVATE
        op_host/circular_pad.cpp
        op_host/circular_pad_common.cc
)

target_include_directories(opsproto PRIVATE
        $CMAKECURRENTSOURCEDIR/ophost
        ${CMAKE_SOURCE_DIR}/src/common/inc
        $ASCENDCANNPACKAGEPATH/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        $ASCENDCANNPACKAGEPATH/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        $ASCENDCANNPACKAGEPATH/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

#手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/circular_pad_def.cpp
)

target_include_directories(op_host_aclnnInner PRIVATE
        $CMAKECURRENTSOURCEDIR/ophost
        ${CMAKE_SOURCE_DIR}/src/common/inc
        $ASCENDCANNPACKAGEPATH/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        $ASCENDCANNPACKAGEPATH/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        $ASCENDCANNPACKAGEPATH/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

target_sources(opapi PRIVATE
        op_host/padv3.cpp
        op_host/aclnn_circular_pad2d.cpp
)

install(FILES op_kernel/circular_pad.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/circular_pad_2d.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/circular_pad_3d.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/circular_pad_common.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/circular_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_host/aclnn_circular_pad2d.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)