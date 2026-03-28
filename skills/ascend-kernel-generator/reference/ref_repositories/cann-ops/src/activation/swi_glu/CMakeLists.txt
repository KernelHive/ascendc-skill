add_ops_compile_options(
        OP_NAME SwiGlu
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/swi_glu_def.cpp
)

# optiling
target_sources(optiling PRIVATE
        op_host/swi_glu.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/swi_glu_tiling.h
)


# opproto
target_sources(opsproto PRIVATE
        op_host/swi_glu_def.cpp
)

install(FILES op_kernel/swi_glu.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/swi_glu_bf16.hpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/swi_glu_impl.hpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/swi_glu_single.hpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/glu_tiling_kernel.hpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/glu_tiling.hpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

# target_include_directories(optiling PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/op_host
#         ${CMAKE_SOURCE_DIR}/src/common/inc
#         ${ASCEND_CANN_PACKAGE_PATH}/include
#         ${ASCEND_CANN_PACKAGE_PATH}/include/external
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
# )

# target_include_directories(opsproto PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/op_host
#         ${CMAKE_SOURCE_DIR}/src/common/inc
#         ${ASCEND_CANN_PACKAGE_PATH}/include
#         ${ASCEND_CANN_PACKAGE_PATH}/include/external
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
# )

# target_include_directories(op_host_aclnnInner PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/op_host
#         ${CMAKE_SOURCE_DIR}/src/common/inc
#         ${ASCEND_CANN_PACKAGE_PATH}/include
#         ${ASCEND_CANN_PACKAGE_PATH}/include/external
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
#         ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
# )

# target_sources(opapi PRIVATE
#         # op_host/
#         # op_host/
# )


