add_ops_compile_options(
        OP_NAME GeluQuant
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
op_host/gelu_quant_def.cpp
)

target_sources(optiling PRIVATE
        op_host/gelu_quant.cpp
)
target_sources(opsproto PRIVATE
        op_host/gelu_quant_def.cpp
)
target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/gelu_quant_tiling.h
)


install(DIRECTORY op_kernel/
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic
        FILES_MATCHING PATTERN "*.cpp")

install(DIRECTORY op_kernel/
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic
        FILES_MATCHING PATTERN "*.h")