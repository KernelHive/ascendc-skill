add_ops_compile_options(
        OP_NAME DiagFlat
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

# target_sources(op_host_aclnn PRIVATE
# op_host/add_custom.cpp
# )

#tiling
target_sources(optiling PRIVATE
        op_host/diag_flat.cpp
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

#proto 文件
target_sources(opsproto PRIVATE
        op_host/diag_flat_ops.cc
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


# 信息库文件
target_sources(op_host_aclnnInner PRIVATE
        op_host/diag_flat_def.cpp
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
        ../../math/diag_v2/op_host/diag_flat_l0.cpp
        op_host/aclnn_diag_flat.cpp
)

# kernel 侧文件
install(FILES op_kernel/diag_flat.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/diag_flat_nd_to_2d_b16_less.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/diag_flat_nd_to_2d_with_few.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        
install(FILES op_kernel/diag_flat_nd_to_2d_b16_more64.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        
install(FILES op_kernel/diag_flat_nd_to_2d.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        
# aclnn L2文件
install(FILES op_host/aclnn_diag_flat.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)