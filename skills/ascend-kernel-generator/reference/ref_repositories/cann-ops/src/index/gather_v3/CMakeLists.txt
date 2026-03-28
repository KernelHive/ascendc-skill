add_ops_compile_options(
        OP_NAME GatherV3
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)


# optiling
target_sources(optiling PRIVATE
        op_host/gather_v3.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)


# opproto
target_sources(opsproto PRIVATE
        op_host/gather_v3_def.cpp
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)


# 手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/gather_v3_def.cpp
)

target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

target_sources(opapi PRIVATE
        op_host/gather_v3_l0.cpp
        op_host/aclnn_index_select.cpp
)


install(FILES op_kernel/gather_v3.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/gather_v3_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/gather_v3_tmpl_0.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/gather_v3_tmpl_1.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/gather_v3_tmpl_2.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/gather_v3_tmpl_3.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_host/aclnn_index_select.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)

add_subdirectory(framework)