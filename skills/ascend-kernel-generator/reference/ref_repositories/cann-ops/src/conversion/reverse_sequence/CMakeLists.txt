add_ops_compile_options(
    OP_NAME ReverseSequence
    OPTIONS --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

#自动生成的aclnn
target_sources(op_host_aclnn PRIVATE
op_host/reverse_sequence.cpp
)

#optiling
target_sources(optiling PRIVATE
    op_host/reverse_sequence.cpp
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
    op_host/reverse_sequence.cpp
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

install(FILES op_kernel/reverse_sequence.cpp
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/reverse_sequence_base.h
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/reverse_sequence_batch_0.h
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/reverse_sequence_batch_1.h
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/platform.h
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)