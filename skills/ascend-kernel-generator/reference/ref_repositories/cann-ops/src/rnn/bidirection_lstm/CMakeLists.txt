add_ops_compile_options(
        OP_NAME BidirectionLSTM
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/bidirection_lstm.cpp
)

target_sources(optiling PRIVATE
        op_host/bidirection_lstm.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
        # op_host/angle_v2.cpp
)

install(FILES op_kernel/bidirection_lstm.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/lstm_bidir_fp16.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/lstm_bidir_fp16.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

