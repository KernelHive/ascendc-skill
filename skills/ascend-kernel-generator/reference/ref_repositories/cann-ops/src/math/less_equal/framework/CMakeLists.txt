if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mindspore")
    if(EXISTS  "${CMAKE_CURRENT_SOURCE_DIR}/caffe_plugin")
        add_subdirectory(caffe_plugin)
    endif()
    if(EXISTS  "${CMAKE_CURRENT_SOURCE_DIR}/tf_plugin")
        add_subdirectory(tf_plugin)
    endif()
    if(EXISTS  "${CMAKE_CURRENT_SOURCE_DIR}/onnx_plugin")
        add_subdirectory(onnx_plugin)
    endif()
endif()

# generate npu_supported_ops.json
add_npu_support_target(TARGET npu_supported_ops
        OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
        OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core
        INSTALL_DIR packages/vendors/${vendor_name}/framework/${ASCEND_FRAMEWORK_TYPE}
)