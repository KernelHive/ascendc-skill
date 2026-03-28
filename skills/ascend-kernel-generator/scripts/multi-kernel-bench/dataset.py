category2exampleop = {
    "matmul": "matmul_add",
    "activation": "leaky_relu",
    "loss": "mse_loss",
    "normalization": "layer_norm",
    'reduce': 'reduce_sum'
}
dataset = {
    "log_softmax": {
        "category": "activation"
    },
    "softsign": {
        "category": "activation"
    },
    "relu": {
        "category": "activation"
    },
    "elu": {
        "category": "activation"
    },
    "softplus": {
        "category": "activation"
    },
    "softmax": {
        "category": "activation"
    },
    "selu": {
        "category": "activation"
    },
    "min_gpt_new_gelu": {
        "category": "activation"
    },
    "gelu": {
        "category": "activation"
    },
    "tanh": {
        "category": "activation"
    },
    "sigmoid": {
        "category": "activation"
    },
    "hardsigmoid": {
        "category": "activation"
    },
    "swish": {
        "category": "activation"
    },
    "leaky_relu": {
        "category": "activation"
    },
    "hardtanh": {
        "category": "activation"
    },
    "where_broadcast": {
        "category": "broadcast"
    },
    "logic_and_broadcast": {
        "category": "broadcast"
    },
    "power_broadcast": {
        "category": "broadcast"
    },
    "max_broadcast": {
        "category": "broadcast"
    },
    "clamp_broadcast": {
        "category": "broadcast"
    },
    "add_bias_broadcast": {
        "category": "broadcast"
    },
    "add_bias_four_dim_broadcast": {
        "category": "broadcast"
    },
    "elmentwise_mul_broadcast": {
        "category": "broadcast"
    },
    "division_broadcast": {
        "category": "broadcast"
    },
    "subtract_with_bias_broadcast": {
        "category": "broadcast"
    },
    "conv_standard_3d_square_input_square_kernel": {
        "category": "convolution"
    },
    "conv_transposed_2d_square_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated": {
        "category": "convolution"
    },
    "conv_transposed_2d_asymmetric_input_square_kernel": {
        "category": "convolution"
    },
    "conv_depthwise_separable_2d": {
        "category": "convolution"
    },
    "conv_standard_2d_square_input_square_kernel": {
        "category": "convolution"
    },
    "conv_transposed_3d_square_input_square_kernel": {
        "category": "convolution"
    },
    "conv_transposed_1d_dilated": {
        "category": "convolution"
    },
    "conv_transposed_1d": {
        "category": "convolution"
    },
    "conv_pointwise_2d": {
        "category": "convolution"
    },
    "conv_standard_2d_square_input_asymmetric_kernel_dilated_padded": {
        "category": "convolution"
    },
    "conv_transposed_2d_asymmetric_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_transposed_2d_asymmetric_input_square_kernel_dilated_padded_strided": {
        "category": "convolution"
    },
    "conv_transposed_2d_square_input_square_kernel": {
        "category": "convolution"
    },
    "conv_transposed_3d_asymmetric_input_square_kernel_strided_padded_grouped": {
        "category": "convolution"
    },
    "conv_transposed_3d_square_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_depthwise_2d_square_input_square_kernel": {
        "category": "convolution"
    },
    "conv_standard_3d_asymmetric_input_square_kernel": {
        "category": "convolution"
    },
    "conv_standard_2d_asymmetric_input_square_kernel": {
        "category": "convolution"
    },
    "conv_standard_2d_square_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated": {
        "category": "convolution"
    },
    "conv_standard_3d_square_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_transposed_3d_asymmetric_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_standard_3d_asymmetric_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_standard_1d_dilated_strided": {
        "category": "convolution"
    },
    "conv_depthwise_2d_asymmetric_input_square_kernel": {
        "category": "convolution"
    },
    "conv_depthwise_2d_asymmetric_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded": {
        "category": "convolution"
    },
    "conv_standard_1d": {
        "category": "convolution"
    },
    "conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped": {
        "category": "convolution"
    },
    "conv_depthwise_2d_square_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_transposed_3d_asymmetric_input_square_kernel": {
        "category": "convolution"
    },
    "conv_transposed_3d_square_input_square_kernel_padded_dilated_strided": {
        "category": "convolution"
    },
    "conv_standard_2d_asymmetric_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "lenet5": {
        "category": "arch"
    },
    "unet_softmax": {
        "category": "arch"
    },
    "vision_attention": {
        "category": "arch"
    },
    "densenet121_transition_layer": {
        "category": "arch"
    },
    "shallow_wide_mlp": {
        "category": "arch"
    },
    "mini_gpt_block": {
        "category": "arch"
    },
    "mamba_return_final_state": {
        "category": "arch"
    },
    "shufflenet_unit": {
        "category": "arch"
    },
    "swin_mlp": {
        "category": "arch"
    },
    "ltsm_cn": {
        "category": "arch"
    },
    "ltsm_hn": {
        "category": "arch"
    },
    "vision_transformer": {
        "category": "arch"
    },
    "net_vlad_with_ghost_clusters": {
        "category": "arch"
    },
    "net_vlad_no_ghost_clusters": {
        "category": "arch"
    },
    "relu_self_attention": {
        "category": "arch"
    },
    "mobilenet_v1": {
        "category": "arch"
    },
    "mamba_return_y": {
        "category": "arch"
    },
    "regnet": {
        "category": "arch"
    },
    "mlp": {
        "category": "arch"
    },
    "efficientnet_mb_conv": {
        "category": "arch"
    },
    "densenet121_dense_block": {
        "category": "arch"
    },
    "swintransformer_v2": {
        "category": "arch"
    },
    "densenet121": {
        "category": "arch"
    },
    "googlenet_inception_v1": {
        "category": "arch"
    },
    "squeeze_net_fire_module": {
        "category": "arch"
    },
    "resnet101": {
        "category": "arch"
    },
    "squeeze_net": {
        "category": "arch"
    },
    "vgg16": {
        "category": "arch"
    },
    "efficientnet_b0": {
        "category": "arch"
    },
    "ltsm_bidirectional": {
        "category": "arch"
    },
    "deep_narrow_mlp": {
        "category": "arch"
    },
    "gru": {
        "category": "arch"
    },
    "resnet18": {
        "category": "arch"
    },
    "ltsm": {
        "category": "arch"
    },
    "vanilla_rnn_hidden": {
        "category": "arch"
    },
    "googlenet_inception_module": {
        "category": "arch"
    },
    "efficientnet_b2": {
        "category": "arch"
    },
    "gru_bidirectional_hidden": {
        "category": "arch"
    },
    "shufflenet": {
        "category": "arch"
    },
    "gru_birectional": {
        "category": "arch"
    },
    "mobilenet_v2": {
        "category": "arch"
    },
    "vgg19": {
        "category": "arch"
    },
    "alexnet": {
        "category": "arch"
    },
    "gru_hidden": {
        "category": "arch"
    },
    "efficientnet_b1": {
        "category": "arch"
    },
    "min_gpt_causal_attention": {
        "category": "arch"
    },
    "vanilla_rnn": {
        "category": "arch"
    },
    "densenet201": {
        "category": "arch"
    },
    "resnet_basic_block": {
        "category": "arch"
    },
    "convolutional_vision_transformer": {
        "category": "arch"
    },
    "convtranspose3d_relu_groupnorm": {
        "category": "fuse"
    },
    "conv2d_subtract_hard_swish_max_pool_mish": {
        "category": "fuse"
    },
    "conv_transpose3d_batch_norm_avg_pool_avg_pool": {
        "category": "fuse"
    },
    "conv3d_divide_max_global_avg_pool_bias_add_sum": {
        "category": "fuse"
    },
    "gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu": {
        "category": "fuse"
    },
    "conv3d_hardswish_relu_softmax_mean": {
        "category": "fuse"
    },
    "conv2d_min_add_multiply": {
        "category": "fuse"
    },
    "conv_transpose2d_gelu_group_norm": {
        "category": "fuse"
    },
    "conv_transpose2d_add_min_gelu_multiply": {
        "category": "fuse"
    },
    "matmul_divide_gelu": {
        "category": "fuse"
    },
    "conv2d_relu_hard_swish": {
        "category": "fuse"
    },
    "conv2d_tanh_scaling_bias_add_max": {
        "category": "fuse"
    },
    "conv_transpose3d_multiply_max_global_avg_pool_clamp": {
        "category": "fuse"
    },
    "matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp": {
        "category": "fuse"
    },
    "gemm_divide_sum_scaling": {
        "category": "fuse"
    },
    "conv2d_batch_norm_scaling": {
        "category": "fuse"
    },
    "conv_transpose3d_avg_pool_clamp_softmax_multiply": {
        "category": "fuse"
    },
    "conv_transpose2d_bias_add_clamp_scaling_clamp_divide": {
        "category": "fuse"
    },
    "conv3d_multiply_instance_norm_clamp_multiply_max": {
        "category": "fuse"
    },
    "conv_transpose3d_layer_norm_gelu_scaling": {
        "category": "fuse"
    },
    "conv3d_group_norm_min_clamp_dropout": {
        "category": "fuse"
    },
    "gemm_group_norm_swish_multiply_swish": {
        "category": "fuse"
    },
    "conv2d_subtract_tanh_subtract_avg_pool": {
        "category": "fuse"
    },
    "matmul_swish_scaling": {
        "category": "fuse"
    },
    "conv2d_gelu_global_avg_pool": {
        "category": "fuse"
    },
    "matmul_min_subtract": {
        "category": "fuse"
    },
    "conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean": {
        "category": "fuse"
    },
    "conv_transpose2d_max_pool_hardtanh_mean_tanh": {
        "category": "fuse"
    },
    "matmul_subtract_multiply_relu": {
        "category": "fuse"
    },
    "conv_transpose2d_subtract_tanh": {
        "category": "fuse"
    },
    "matmul_swish_sum_group_norm": {
        "category": "fuse"
    },
    "conv3d_max_log_sum_exp_relu": {
        "category": "fuse"
    },
    "conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp": {
        "category": "fuse"
    },
    "conv_transpose3d_sum_layer_norm_avg_pool_gelu": {
        "category": "fuse"
    },
    "conv2d_avg_pool_sigmoid_sum": {
        "category": "fuse"
    },
    "conv3d_relu_leaky_relu_gelu_sigmoid_bias_add": {
        "category": "fuse"
    },
    "gemm_group_norm_min_bias_add": {
        "category": "fuse"
    },
    "conv3d_softmax_max_pool_max_pool": {
        "category": "fuse"
    },
    "gemm_group_norm_hardtanh": {
        "category": "fuse"
    },
    "conv2d_group_norm_scale_max_pool_clamp": {
        "category": "fuse"
    },
    "gemm_swish_divide_clamp_tanh_clamp": {
        "category": "fuse"
    },
    "convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply": {
        "category": "fuse"
    },
    "conv2d_divide_leaky_relu": {
        "category": "fuse"
    },
    "matmul_dropout_mean_softmax": {
        "category": "fuse"
    },
    "conv_transpose3d_swish_group_norm_hard_swish": {
        "category": "fuse"
    },
    "conv2d_instance_norm_divide": {
        "category": "fuse"
    },
    "conv2d_scaling_min": {
        "category": "fuse"
    },
    "conv_transpose3d_scaling_avg_pool_bias_add_scaling": {
        "category": "fuse"
    },
    "gemm_max_subtract_gelu": {
        "category": "fuse"
    },
    "gemm_batch_norm_scaling_softmax": {
        "category": "fuse"
    },
    "conv2d_multiply_leaky_relu_gelu": {
        "category": "fuse"
    },
    "conv_transpose3d_batch_norm_subtract": {
        "category": "fuse"
    },
    "convtranspose2d_batchnorm_tanh_maxpool_groupnorm": {
        "category": "fuse"
    },
    "conv2d_activation_batch_norm": {
        "category": "fuse"
    },
    "gemm_scale_batch_norm": {
        "category": "fuse"
    },
    "conv_transpose3d_sum_residual_add_multiply_residual_add": {
        "category": "fuse"
    },
    "conv_transpose3d_clamp_min_divide": {
        "category": "fuse"
    },
    "gemm_scaling_hard_tanh_gelu": {
        "category": "fuse"
    },
    "matmul_scale_residual_add_clamp_log_sum_exp_mish": {
        "category": "fuse"
    },
    "matmul_scaling_residual_add": {
        "category": "fuse"
    },
    "bmm_instance_norm_sum_residual_add_multiply": {
        "category": "fuse"
    },
    "conv_transpose3d_max_pool_softmax_subtract_swish_max": {
        "category": "fuse"
    },
    "conv2d_mish_mish": {
        "category": "fuse"
    },
    "gemm_add_relu": {
        "category": "fuse"
    },
    "gemm_relu_divide": {
        "category": "fuse"
    },
    "conv3d_leaky_relu_sum_clamp_gelu": {
        "category": "fuse"
    },
    "matmul_group_norm_leaky_relu_sum": {
        "category": "fuse"
    },
    "conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max": {
        "category": "fuse"
    },
    "gemm_sigmoid_sum_log_sum_exp": {
        "category": "fuse"
    },
    "conv_transpose3d_add_hard_swish": {
        "category": "fuse"
    },
    "conv2d_min_tanh_tanh": {
        "category": "fuse"
    },
    "conv2d_relu_bias_add": {
        "category": "fuse"
    },
    "gemm_scale_batchnorm": {
        "category": "fuse"
    },
    "gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add": {
        "category": "fuse"
    },
    "matmul_add_swish_tanh_gelu_hardtanh": {
        "category": "fuse"
    },
    "matmul_sigmoid_sum": {
        "category": "fuse"
    },
    "conv3d_mish_tanh": {
        "category": "fuse"
    },
    "matmul_batch_norm_bias_add_divide_swish": {
        "category": "fuse"
    },
    "conv2d_subtract_subtract_mish": {
        "category": "fuse"
    },
    "conv_transpose3d_scale_batch_norm_global_avg_pool": {
        "category": "fuse"
    },
    "gemm_bias_add_hardtanh_mish_group_norm": {
        "category": "fuse"
    },
    "conv3d_scaling_tanh_multiply_sigmoid": {
        "category": "fuse"
    },
    "conv_transpose3d_softmax_sigmoid": {
        "category": "fuse"
    },
    "matmul_gelu_softmax": {
        "category": "fuse"
    },
    "conv2d_add_scale_sigmoid_group_norm": {
        "category": "fuse"
    },
    "matmul_avg_pool_gelu_scale_max": {
        "category": "fuse"
    },
    "convtranspose2d_softmax_biasadd_scaling_sigmoid": {
        "category": "fuse"
    },
    "matmul_mish_mish": {
        "category": "fuse"
    },
    "matmul_max_pool_sum_scale": {
        "category": "fuse"
    },
    "conv_transpose3d_leaky_relu_multiply_leaky_relu_max": {
        "category": "fuse"
    },
    "gemm_multiply_leakyrelu": {
        "category": "fuse"
    },
    "gemm_sigmoid_scaling_residual_add": {
        "category": "fuse"
    },
    "conv_transpose2d_mish_add_hardtanh_scaling": {
        "category": "fuse"
    },
    "gemm_batch_norm_gelu_group_norm_mean_relu": {
        "category": "fuse"
    },
    "conv3d_group_norm_mean": {
        "category": "fuse"
    },
    "conv2d_hard_swish_relu": {
        "category": "fuse"
    },
    "convtranspose3d_mean_add_softmax_tanh_scaling": {
        "category": "fuse"
    },
    "conv_transpose3d_max_max_sum": {
        "category": "fuse"
    },
    "conv_transpose2d_min_sum_gelu_add": {
        "category": "fuse"
    },
    "conv3d_min_softmax": {
        "category": "fuse"
    },
    "triplet_margin_loss": {
        "category": "loss"
    },
    "kl_div_loss": {
        "category": "loss"
    },
    "cosine_similarity_loss": {
        "category": "loss"
    },
    "huber_loss": {
        "category": "loss"
    },
    "mse_loss": {
        "category": "loss"
    },
    "cross_entropy_loss": {
        "category": "loss"
    },
    "hinge_loss": {
        "category": "loss"
    },
    "cumsum_exclusive": {
        "category": "math"
    },
    "cumprod": {
        "category": "math"
    },
    "masked_cumsum": {
        "category": "math"
    },
    "matrix_scalar_multiplication": {
        "category": "math"
    },
    "cumsum": {
        "category": "math"
    },
    "cumsum_reverse": {
        "category": "math"
    },
    "matmul_with_diagonal_matrices": {
        "category": "matmul"
    },
    "matmul_with_transposed_a": {
        "category": "matmul"
    },
    "matmul_for_lower_triangular_matrices": {
        "category": "matmul"
    },
    "batched_matrix_multiplication": {
        "category": "matmul"
    },
    "square_matrix_multiplication": {
        "category": "matmul"
    },
    "matmul_with_irregular_shapes": {
        "category": "matmul"
    },
    "four_dim_tensor_matrix_multiplication": {
        "category": "matmul"
    },
    "tall_skinny_matrix_multiplication": {
        "category": "matmul"
    },
    "three_dim_tensor_matrix_multiplication": {
        "category": "matmul"
    },
    "matmul_with_large_k_dimension": {
        "category": "matmul"
    },
    "matmul_with_small_k_dimension": {
        "category": "matmul"
    },
    "standard_matrix_multiplication": {
        "category": "matmul"
    },
    "matmul_with_transposed_b": {
        "category": "matmul"
    },
    "matrix_vector_multiplication": {
        "category": "matmul"
    },
    "matmul_with_transposed_both": {
        "category": "matmul"
    },
    "matmul_for_symmetric_matrices": {
        "category": "matmul"
    },
    "matmul_for_upper_triangular_matrices": {
        "category": "matmul"
    },
    "rms_norm": {
        "category": "normalization"
    },
    "l2_norm": {
        "category": "normalization"
    },
    "l1_norm": {
        "category": "normalization"
    },
    "frobenius_norm": {
        "category": "normalization"
    },
    "group_norm": {
        "category": "normalization"
    },
    "batch_norm": {
        "category": "normalization"
    },
    "instance_norm": {
        "category": "normalization"
    },
    "layer_norm": {
        "category": "normalization"
    },
    "adam": {
        "category": "optimizer"
    },
    "adagrad": {
        "category": "optimizer"
    },
    "lamb": {
        "category": "optimizer"
    },
    "rmsprop": {
        "category": "optimizer"
    },
    "sgd": {
        "category": "optimizer"
    },
    "average_pooling_3d": {
        "category": "pooling"
    },
    "max_pooling_1d": {
        "category": "pooling"
    },
    "max_pooling_3d": {
        "category": "pooling"
    },
    "average_pooling_2d": {
        "category": "pooling"
    },
    "average_pooling_1d": {
        "category": "pooling"
    },
    "average_pooling1d": {
        "category": "pooling"
    },
    "max_pooling_2d": {
        "category": "pooling"
    },
    "index_select": {
        "category": "index"
    },
    "inplace_update": {
        "category": "index"
    },
    "argmax_over_a_dimension": {
        "category": "index"
    },
    "gather": {
        "category": "index"
    },
    "scatter": {
        "category": "index"
    },
    "index_copy": {
        "category": "index"
    },
    "masked_fill": {
        "category": "index"
    },
    "index_add": {
        "category": "index"
    },
    "embedding": {
        "category": "index"
    },
    "scatter_add": {
        "category": "index"
    },
    "take_along_dim": {
        "category": "index"
    },
    "argmin_over_a_dimension": {
        "category": "index"
    },
    "bicubic_upsample": {
        "category": "resize"
    },
    "nearest_neighbor_upsample": {
        "category": "resize"
    },
    "downsample_bilinear": {
        "category": "resize"
    },
    "resize_with_antialias": {
        "category": "resize"
    },
    "interpolate_dynamic": {
        "category": "resize"
    },
    "upsample_grid_sample": {
        "category": "resize"
    },
    "bilinear_upsample": {
        "category": "resize"
    },
    "grid_sample_random_warp": {
        "category": "resize"
    },
    "grid_sample_affine": {
        "category": "resize"
    },
    "trilinear_upsample": {
        "category": "resize"
    },
    "sum_reduction_over_a_dimension": {
        "category": "reduce"
    },
    "product_reduction_over_a_dimension": {
        "category": "reduce"
    },
    "mean_reduction_over_a_dimension": {
        "category": "reduce"
    },
    "max_reduction_over_a_dimension": {
        "category": "reduce"
    },
    "min_reduction_over_a_dimension": {
        "category": "reduce"
    },
    "causal_attention": {
        "category": "attention"
    },
    "kv_cached_attention_inference": {
        "category": "attention"
    },
    "multi_head_attention": {
        "category": "attention"
    },
    "scaled_dot_product_attention_long_context": {
        "category": "attention"
    },
    "cross_attention": {
        "category": "attention"
    },
    "kv_cached_chat_batch_attention": {
        "category": "attention"
    },
    "multi_query_attention": {
        "category": "attention"
    },
    "sparse_attention": {
        "category": "attention"
    },
    "cross_modal_attention": {
        "category": "attention"
    },
    "kv_cached_speculative_attention": {
        "category": "attention"
    },
    "scaled_dot_product_attention": {
        "category": "attention"
    },
    "windowed_causal_attention": {
        "category": "attention"
    },
    "group_query_attention": {
        "category": "attention"
    },
    "linear_attention": {
        "category": "attention"
    },
    "scaled_dot_product_attention_inference": {
        "category": "attention"
    }
}


level = {
  "upsample_grid_sample": {
    "category": "resize",
    "level": "level1"
  },
  "trilinear_upsample": {
    "category": "resize",
    "level": "level1"
  },
  "grid_sample_affine": {
    "category": "resize",
    "level": "level1"
  },
  "bicubic_upsample": {
    "category": "resize",
    "level": "level1"
  },
  "grid_sample_random_warp": {
    "category": "resize",
    "level": "level1"
  },
  "downsample_bilinear": {
    "category": "resize",
    "level": "level1"
  },
  "bilinear_upsample": {
    "category": "resize",
    "level": "level1"
  },
  "nearest_neighbor_upsample": {
    "category": "resize",
    "level": "level1"
  },
  "interpolate_dynamic": {
    "category": "resize",
    "level": "level1"
  },
  "resize_with_antialias": {
    "category": "resize",
    "level": "level1"
  },
  "average_pooling_1d": {
    "category": "pooling",
    "level": "level1"
  },
  "average_pooling_2d": {
    "category": "pooling",
    "level": "level1"
  },
  "average_pooling_3d": {
    "category": "pooling",
    "level": "level1"
  },
  "max_pooling_3d": {
    "category": "pooling",
    "level": "level1"
  },
  "max_pooling_2d": {
    "category": "pooling",
    "level": "level1"
  },
  "max_pooling_1d": {
    "category": "pooling",
    "level": "level1"
  },
  "batched_matrix_multiplication": {
    "category": "matmul",
    "level": "level1"
  },
  "matmul_with_diagonal_matrices": {
    "category": "matmul",
    "level": "level1"
  },
  "four_dim_tensor_matrix_multiplication": {
    "category": "matmul",
    "level": "level1"
  },
  "matmul_with_small_k_dimension": {
    "category": "matmul",
    "level": "level1"
  },
  "matmul_for_symmetric_matrices": {
    "category": "matmul",
    "level": "level1"
  },
  "matmul_with_large_k_dimension": {
    "category": "matmul",
    "level": "level1"
  },
  "matmul_with_transposed_b": {
    "category": "matmul",
    "level": "level1"
  },
  "matmul_with_transposed_a": {
    "category": "matmul",
    "level": "level1"
  },
  "matmul_for_lower_triangular_matrices": {
    "category": "matmul",
    "level": "level1"
  },
  "three_dim_tensor_matrix_multiplication": {
    "category": "matmul",
    "level": "level1"
  },
  "square_matrix_multiplication": {
    "category": "matmul",
    "level": "level1"
  },
  "matmul_with_transposed_both": {
    "category": "matmul",
    "level": "level1"
  },
  "matmul_with_irregular_shapes": {
    "category": "matmul",
    "level": "level1"
  },
  "matmul_for_upper_triangular_matrices": {
    "category": "matmul",
    "level": "level1"
  },
  "tall_skinny_matrix_multiplication": {
    "category": "matmul",
    "level": "level1"
  },
  "standard_matrix_multiplication": {
    "category": "matmul",
    "level": "level1"
  },
  "matrix_vector_multiplication": {
    "category": "matmul",
    "level": "level1"
  },
  "index_copy": {
    "category": "index",
    "level": "level1"
  },
  "scatter_add": {
    "category": "index",
    "level": "level1"
  },
  "gather": {
    "category": "index",
    "level": "level1"
  },
  "index_select": {
    "category": "index",
    "level": "level1"
  },
  "masked_fill": {
    "category": "index",
    "level": "level1"
  },
  "index_add": {
    "category": "index",
    "level": "level1"
  },
  "scatter": {
    "category": "index",
    "level": "level1"
  },
  "argmax_over_a_dimension": {
    "category": "index",
    "level": "level1"
  },
  "take_along_dim": {
    "category": "index",
    "level": "level1"
  },
  "inplace_update": {
    "category": "index",
    "level": "level1"
  },
  "embedding": {
    "category": "index",
    "level": "level1"
  },
  "argmin_over_a_dimension": {
    "category": "index",
    "level": "level1"
  },
  "matmul_gelu_softmax": {
    "category": "fuse",
    "level": "level2"
  },
  "conv3d_multiply_instance_norm_clamp_multiply_max": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_subtract_tanh_subtract_avg_pool": {
    "category": "fuse",
    "level": "level2"
  },
  "conv3d_divide_max_global_avg_pool_bias_add_sum": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_group_norm_scale_max_pool_clamp": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_activation_batch_norm": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_sum_layer_norm_avg_pool_gelu": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_group_norm_min_bias_add": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_batch_norm_scaling_softmax": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_add_hard_swish": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_max_max_sum": {
    "category": "fuse",
    "level": "level2"
  },
  "convtranspose3d_relu_groupnorm": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_scale_residual_add_clamp_log_sum_exp_mish": {
    "category": "fuse",
    "level": "level2"
  },
  "convtranspose2d_softmax_biasadd_scaling_sigmoid": {
    "category": "fuse",
    "level": "level2"
  },
  "bmm_instance_norm_sum_residual_add_multiply": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_scaling_avg_pool_bias_add_scaling": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_group_norm_hardtanh": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_max_pool_sum_scale": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_batch_norm_subtract": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_batch_norm_gelu_group_norm_mean_relu": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_min_tanh_tanh": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_swish_group_norm_hard_swish": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_swish_divide_clamp_tanh_clamp": {
    "category": "fuse",
    "level": "level2"
  },
  "convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_batch_norm_avg_pool_avg_pool": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_swish_scaling": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_max_subtract_gelu": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_multiply_leakyrelu": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose2d_add_min_gelu_multiply": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_group_norm_leaky_relu_sum": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_sigmoid_scaling_residual_add": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_dropout_mean_softmax": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_softmax_sigmoid": {
    "category": "fuse",
    "level": "level2"
  },
  "conv3d_max_log_sum_exp_relu": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_subtract_multiply_relu": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose2d_max_pool_hardtanh_mean_tanh": {
    "category": "fuse",
    "level": "level2"
  },
  "conv3d_softmax_max_pool_max_pool": {
    "category": "fuse",
    "level": "level2"
  },
  "conv3d_scaling_tanh_multiply_sigmoid": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose2d_gelu_group_norm": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose2d_subtract_tanh": {
    "category": "fuse",
    "level": "level2"
  },
  "conv3d_leaky_relu_sum_clamp_gelu": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_subtract_subtract_mish": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_tanh_scaling_bias_add_max": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_divide_leaky_relu": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_add_relu": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_divide_sum_scaling": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_divide_gelu": {
    "category": "fuse",
    "level": "level2"
  },
  "convtranspose2d_batchnorm_tanh_maxpool_groupnorm": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose2d_bias_add_clamp_scaling_clamp_divide": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_multiply_leaky_relu_gelu": {
    "category": "fuse",
    "level": "level2"
  },
  "conv3d_relu_leaky_relu_gelu_sigmoid_bias_add": {
    "category": "fuse",
    "level": "level2"
  },
  "conv3d_mish_tanh": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_add_scale_sigmoid_group_norm": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_swish_sum_group_norm": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_sigmoid_sum": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_batch_norm_bias_add_divide_swish": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_min_add_multiply": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_mish_mish": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_scaling_hard_tanh_gelu": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_sum_residual_add_multiply_residual_add": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_instance_norm_divide": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp": {
    "category": "fuse",
    "level": "level2"
  },
  "conv3d_min_softmax": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_batch_norm_scaling": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose2d_min_sum_gelu_add": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_layer_norm_gelu_scaling": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_scale_batch_norm": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_group_norm_swish_multiply_swish": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_avg_pool_clamp_softmax_multiply": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_min_subtract": {
    "category": "fuse",
    "level": "level2"
  },
  "conv3d_group_norm_mean": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_avg_pool_sigmoid_sum": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_relu_divide": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_mish_mish": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_scale_batchnorm": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_add_swish_tanh_gelu_hardtanh": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_leaky_relu_multiply_leaky_relu_max": {
    "category": "fuse",
    "level": "level2"
  },
  "conv3d_hardswish_relu_softmax_mean": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose2d_mish_add_hardtanh_scaling": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_clamp_min_divide": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_multiply_max_global_avg_pool_clamp": {
    "category": "fuse",
    "level": "level2"
  },
  "conv3d_group_norm_min_clamp_dropout": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_hard_swish_relu": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_max_pool_softmax_subtract_swish_max": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_gelu_global_avg_pool": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_sigmoid_sum_log_sum_exp": {
    "category": "fuse",
    "level": "level2"
  },
  "convtranspose3d_mean_add_softmax_tanh_scaling": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_scaling_min": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_relu_hard_swish": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_relu_bias_add": {
    "category": "fuse",
    "level": "level2"
  },
  "conv_transpose3d_scale_batch_norm_global_avg_pool": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_bias_add_hardtanh_mish_group_norm": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_subtract_hard_swish_max_pool_mish": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_avg_pool_gelu_scale_max": {
    "category": "fuse",
    "level": "level2"
  },
  "matmul_scaling_residual_add": {
    "category": "fuse",
    "level": "level2"
  },
  "selu": {
    "category": "activation",
    "level": "level1"
  },
  "swish": {
    "category": "activation",
    "level": "level1"
  },
  "leaky_relu": {
    "category": "activation",
    "level": "level1"
  },
  "min_gpt_new_gelu": {
    "category": "activation",
    "level": "level1"
  },
  "tanh": {
    "category": "activation",
    "level": "level1"
  },
  "softplus": {
    "category": "activation",
    "level": "level1"
  },
  "softsign": {
    "category": "activation",
    "level": "level1"
  },
  "log_softmax": {
    "category": "activation",
    "level": "level1"
  },
  "hardsigmoid": {
    "category": "activation",
    "level": "level1"
  },
  "softmax": {
    "category": "activation",
    "level": "level1"
  },
  "relu": {
    "category": "activation",
    "level": "level1"
  },
  "hardtanh": {
    "category": "activation",
    "level": "level1"
  },
  "sigmoid": {
    "category": "activation",
    "level": "level1"
  },
  "elu": {
    "category": "activation",
    "level": "level1"
  },
  "gelu": {
    "category": "activation",
    "level": "level1"
  },
  "conv_transposed_2d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_3d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_3d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_depthwise_2d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_depthwise_2d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_standard_2d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_depthwise_2d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_2d_asymmetric_input_square_kernel_dilated_padded_strided": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_standard_1d": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_standard_3d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_standard_2d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_3d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_1d_dilated": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_2d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_2d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_3d_square_input_square_kernel_padded_dilated_strided": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_depthwise_separable_2d": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_standard_2d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_standard_2d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_depthwise_2d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_pointwise_2d": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_2d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_standard_3d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_3d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_standard_3d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_standard_1d_dilated_strided": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_standard_3d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_3d_asymmetric_input_square_kernel_strided_padded_grouped": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_transposed_1d": {
    "category": "convolution",
    "level": "level1"
  },
  "conv_standard_2d_square_input_asymmetric_kernel_dilated_padded": {
    "category": "convolution",
    "level": "level1"
  },
  "sgd": {
    "category": "optimizer",
    "level": "level1"
  },
  "rmsprop": {
    "category": "optimizer",
    "level": "level1"
  },
  "lamb": {
    "category": "optimizer",
    "level": "level1"
  },
  "adam": {
    "category": "optimizer",
    "level": "level1"
  },
  "adagrad": {
    "category": "optimizer",
    "level": "level1"
  },
  "sum_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1"
  },
  "mean_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1"
  },
  "product_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1"
  },
  "max_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1"
  },
  "min_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1"
  },
  "hinge_loss": {
    "category": "loss",
    "level": "level1"
  },
  "cross_entropy_loss": {
    "category": "loss",
    "level": "level1"
  },
  "triplet_margin_loss": {
    "category": "loss",
    "level": "level1"
  },
  "mse_loss": {
    "category": "loss",
    "level": "level1"
  },
  "kl_div_loss": {
    "category": "loss",
    "level": "level1"
  },
  "cosine_similarity_loss": {
    "category": "loss",
    "level": "level1"
  },
  "huber_loss": {
    "category": "loss",
    "level": "level1"
  },
  "cumprod": {
    "category": "math",
    "level": "level1"
  },
  "cumsum_exclusive": {
    "category": "math",
    "level": "level1"
  },
  "matrix_scalar_multiplication": {
    "category": "math",
    "level": "level1"
  },
  "cumsum_reverse": {
    "category": "math",
    "level": "level1"
  },
  "cumsum": {
    "category": "math",
    "level": "level1"
  },
  "masked_cumsum": {
    "category": "math",
    "level": "level1"
  },
  "division_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "power_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "elmentwise_mul_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "subtract_with_bias_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "add_bias_four_dim_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "logic_and_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "where_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "add_bias_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "max_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "clamp_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "squeeze_net_fire_module": {
    "category": "arch",
    "level": "level3"
  },
  "densenet121_transition_layer": {
    "category": "arch",
    "level": "level3"
  },
  "net_vlad_no_ghost_clusters": {
    "category": "arch",
    "level": "level3"
  },
  "lenet5": {
    "category": "arch",
    "level": "level3"
  },
  "efficientnet_b0": {
    "category": "arch",
    "level": "level3"
  },
  "ltsm_cn": {
    "category": "arch",
    "level": "level3"
  },
  "squeeze_net": {
    "category": "arch",
    "level": "level3"
  },
  "ltsm": {
    "category": "arch",
    "level": "level3"
  },
  "shufflenet": {
    "category": "arch",
    "level": "level3"
  },
  "mobilenet_v1": {
    "category": "arch",
    "level": "level3"
  },
  "shufflenet_unit": {
    "category": "arch",
    "level": "level3"
  },
  "ltsm_hn": {
    "category": "arch",
    "level": "level3"
  },
  "densenet121_dense_block": {
    "category": "arch",
    "level": "level3"
  },
  "mamba_return_y": {
    "category": "arch",
    "level": "level3"
  },
  "vanilla_rnn": {
    "category": "arch",
    "level": "level3"
  },
  "swintransformer_v2": {
    "category": "arch",
    "level": "level3"
  },
  "resnet101": {
    "category": "arch",
    "level": "level3"
  },
  "gru": {
    "category": "arch",
    "level": "level3"
  },
  "mini_gpt_block": {
    "category": "arch",
    "level": "level3"
  },
  "vision_transformer": {
    "category": "arch",
    "level": "level3"
  },
  "ltsm_bidirectional": {
    "category": "arch",
    "level": "level3"
  },
  "vgg19": {
    "category": "arch",
    "level": "level3"
  },
  "min_gpt_causal_attention": {
    "category": "arch",
    "level": "level3"
  },
  "deep_narrow_mlp": {
    "category": "arch",
    "level": "level3"
  },
  "shallow_wide_mlp": {
    "category": "arch",
    "level": "level3"
  },
  "unet_softmax": {
    "category": "arch",
    "level": "level3"
  },
  "vgg16": {
    "category": "arch",
    "level": "level3"
  },
  "gru_bidirectional_hidden": {
    "category": "arch",
    "level": "level3"
  },
  "resnet18": {
    "category": "arch",
    "level": "level3"
  },
  "alexnet": {
    "category": "arch",
    "level": "level3"
  },
  "relu_self_attention": {
    "category": "arch",
    "level": "level3"
  },
  "regnet": {
    "category": "arch",
    "level": "level3"
  },
  "googlenet_inception_module": {
    "category": "arch",
    "level": "level3"
  },
  "vision_attention": {
    "category": "arch",
    "level": "level3"
  },
  "mlp": {
    "category": "arch",
    "level": "level3"
  },
  "densenet121": {
    "category": "arch",
    "level": "level3"
  },
  "swin_mlp": {
    "category": "arch",
    "level": "level3"
  },
  "googlenet_inception_v1": {
    "category": "arch",
    "level": "level3"
  },
  "efficientnet_b2": {
    "category": "arch",
    "level": "level3"
  },
  "resnet_basic_block": {
    "category": "arch",
    "level": "level3"
  },
  "densenet201": {
    "category": "arch",
    "level": "level3"
  },
  "mobilenet_v2": {
    "category": "arch",
    "level": "level3"
  },
  "gru_hidden": {
    "category": "arch",
    "level": "level3"
  },
  "vanilla_rnn_hidden": {
    "category": "arch",
    "level": "level3"
  },
  "net_vlad_with_ghost_clusters": {
    "category": "arch",
    "level": "level3"
  },
  "convolutional_vision_transformer": {
    "category": "arch",
    "level": "level3"
  },
  "mamba_return_final_state": {
    "category": "arch",
    "level": "level3"
  },
  "efficientnet_mb_conv": {
    "category": "arch",
    "level": "level3"
  },
  "gru_birectional": {
    "category": "arch",
    "level": "level3"
  },
  "efficientnet_b1": {
    "category": "arch",
    "level": "level3"
  },
  "cross_attention": {
    "category": "attention",
    "level": "level2"
  },
  "multi_query_attention": {
    "category": "attention",
    "level": "level2"
  },
  "scaled_dot_product_attention_inference": {
    "category": "attention",
    "level": "level2"
  },
  "windowed_causal_attention": {
    "category": "attention",
    "level": "level2"
  },
  "kv_cached_chat_batch_attention": {
    "category": "attention",
    "level": "level2"
  },
  "group_query_attention": {
    "category": "attention",
    "level": "level2"
  },
  "multi_head_attention": {
    "category": "attention",
    "level": "level2"
  },
  "causal_attention": {
    "category": "attention",
    "level": "level2"
  },
  "kv_cached_speculative_attention": {
    "category": "attention",
    "level": "level2"
  },
  "scaled_dot_product_attention_long_context": {
    "category": "attention",
    "level": "level2"
  },
  "kv_cached_attention_inference": {
    "category": "attention",
    "level": "level2"
  },
  "scaled_dot_product_attention": {
    "category": "attention",
    "level": "level2"
  },
  "sparse_attention": {
    "category": "attention",
    "level": "level2"
  },
  "cross_modal_attention": {
    "category": "attention",
    "level": "level2"
  },
  "linear_attention": {
    "category": "attention",
    "level": "level2"
  },
  "layer_norm": {
    "category": "normalization",
    "level": "level1"
  },
  "l1_norm": {
    "category": "normalization",
    "level": "level1"
  },
  "l2_norm": {
    "category": "normalization",
    "level": "level1"
  },
  "group_norm": {
    "category": "normalization",
    "level": "level1"
  },
  "frobenius_norm": {
    "category": "normalization",
    "level": "level1"
  },
  "instance_norm": {
    "category": "normalization",
    "level": "level1"
  },
  "rms_norm": {
    "category": "normalization",
    "level": "level1"
  },
  "batch_norm": {
    "category": "normalization",
    "level": "level1"
  }
}


level_old = {
  "upsample_grid_sample": {
    "category": "resize",
    "level": "level1"
  },
  "trilinear_upsample": {
    "category": "resize",
    "level": "level1"
  },
  "grid_sample_affine": {
    "category": "resize",
    "level": "level1"
  },
  "bicubic_upsample": {
    "category": "resize",
    "level": "level1"
  },
  "grid_sample_random_warp": {
    "category": "resize",
    "level": "level1"
  },
  "downsample_bilinear": {
    "category": "resize",
    "level": "level1"
  },
  "bilinear_upsample": {
    "category": "resize",
    "level": "level1"
  },
  "nearest_neighbor_upsample": {
    "category": "resize",
    "level": "level1"
  },
  "interpolate_dynamic": {
    "category": "resize",
    "level": "level1"
  },
  "resize_with_antialias": {
    "category": "resize",
    "level": "level1"
  },
  "average_pooling_1d": {
    "category": "pooling",
    "level": "level1",
    "matched_ref": "averagepooling1d"
  },
  "average_pooling_2d": {
    "category": "pooling",
    "level": "level1",
    "matched_ref": "averagepooling2d"
  },
  "average_pooling_3d": {
    "category": "pooling",
    "level": "level1",
    "matched_ref": "averagepooling3d"
  },
  "max_pooling_3d": {
    "category": "pooling",
    "level": "level1",
    "matched_ref": "maxpooling3d"
  },
  "max_pooling_2d": {
    "category": "pooling",
    "level": "level1",
    "matched_ref": "maxpooling2d"
  },
  "max_pooling_1d": {
    "category": "pooling",
    "level": "level1",
    "matched_ref": "maxpooling1d"
  },
  "batched_matrix_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "batchedmatrixmultiplication"
  },
  "matmul_with_diagonal_matrices": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithdiagonalmatrices"
  },
  "four_dim_tensor_matrix_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "3dtensormatrixmultiplication"
  },
  "matmul_with_small_k_dimension": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithsmallkdimension"
  },
  "matmul_for_symmetric_matrices": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulforsymmetricmatrices"
  },
  "matmul_with_large_k_dimension": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithlargekdimension"
  },
  "matmul_with_transposed_b": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithtransposedb"
  },
  "matmul_with_transposed_a": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithtransposeda"
  },
  "matmul_for_lower_triangular_matrices": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulforlowertriangularmatrices"
  },
  "three_dim_tensor_matrix_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "3dtensormatrixmultiplication"
  },
  "square_matrix_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "squarematrixmultiplication"
  },
  "matmul_with_transposed_both": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithtransposedboth"
  },
  "matmul_with_irregular_shapes": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithirregularshapes"
  },
  "matmul_for_upper_triangular_matrices": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulforuppertriangularmatrices"
  },
  "tall_skinny_matrix_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "tallskinnymatrixmultiplication"
  },
  "standard_matrix_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "standardmatrixmultiplication"
  },
  "matrix_vector_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matrixvectormultiplication"
  },
  "index_copy": {
    "category": "index",
    "level": "level1"
  },
  "scatter_add": {
    "category": "index",
    "level": "level1"
  },
  "gather": {
    "category": "index",
    "level": "level1"
  },
  "index_select": {
    "category": "index",
    "level": "level1"
  },
  "masked_fill": {
    "category": "index",
    "level": "level1"
  },
  "index_add": {
    "category": "index",
    "level": "level1"
  },
  "scatter": {
    "category": "index",
    "level": "level1"
  },
  "argmax_over_a_dimension": {
    "category": "index",
    "level": "level1",
    "matched_ref": "argmaxoveradimension"
  },
  "take_along_dim": {
    "category": "index",
    "level": "level1"
  },
  "inplace_update": {
    "category": "index",
    "level": "level1"
  },
  "embedding": {
    "category": "index",
    "level": "level1"
  },
  "argmin_over_a_dimension": {
    "category": "index",
    "level": "level1",
    "matched_ref": "argminoveradimension"
  },
  "matmul_gelu_softmax": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulgelusoftmax"
  },
  "conv3d_multiply_instance_norm_clamp_multiply_max": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dmultiplyinstancenormclampmultiplymax"
  },
  "conv2d_subtract_tanh_subtract_avg_pool": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dsubtracttanhsubtractavgpool"
  },
  "conv3d_divide_max_global_avg_pool_bias_add_sum": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3ddividemaxglobalavgpoolbiasaddsum"
  },
  "conv2d_group_norm_scale_max_pool_clamp": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dgroupnormscalemaxpoolclam"
  },
  "conv2d_activation_batch_norm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dactivationbatchnorm"
  },
  "conv_transpose3d_sum_layer_norm_avg_pool_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dsumlayernormavgpoolgelu"
  },
  "gemm_group_norm_min_bias_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmgroupnormminbiasadd"
  },
  "gemm_batch_norm_scaling_softmax": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmbatchnormscalingsoftmax"
  },
  "conv_transpose3d_add_hard_swish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3daddhardswish"
  },
  "conv_transpose3d_max_max_sum": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dmaxmaxsum"
  },
  "convtranspose3d_relu_groupnorm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3drelugroupnorm"
  },
  "matmul_scale_residual_add_clamp_log_sum_exp_mish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulscaleresidualaddclamplogsumexpmish"
  },
  "convtranspose2d_softmax_biasadd_scaling_sigmoid": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dsoftmaxbiasaddscalingsigmoid"
  },
  "bmm_instance_norm_sum_residual_add_multiply": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "bmminstancenormsumresidualaddmultipl"
  },
  "conv_transpose3d_scaling_avg_pool_bias_add_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dscalingavgpoolbiasaddscaling"
  },
  "gemm_group_norm_hardtanh": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmgroupnormhardtanh"
  },
  "matmul_max_pool_sum_scale": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulmaxpoolsumscale"
  },
  "conv_transpose3d_batch_norm_subtract": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dbatchnormsubtract"
  },
  "gemm_batch_norm_gelu_group_norm_mean_relu": {
    "category": "fuse",
    "level": "level2"
  },
  "conv2d_min_tanh_tanh": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dmintanhtanh"
  },
  "conv_transpose3d_swish_group_norm_hard_swish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dswishgroupnormhardswish"
  },
  "gemm_swish_divide_clamp_tanh_clamp": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmswishdivideclamptanhclam"
  },
  "convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dglobalavgpoolbiasaddlogsumexpsummultipl"
  },
  "conv_transpose3d_batch_norm_avg_pool_avg_pool": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dbatchnormavgpoolavgpool"
  },
  "matmul_swish_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulswishscaling"
  },
  "gemm_max_subtract_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmmaxsubtractgelu"
  },
  "gemm_multiply_leakyrelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmmultiplyleakyrelu"
  },
  "conv_transpose2d_add_min_gelu_multiply": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2daddmingelumultipl"
  },
  "matmul_group_norm_leaky_relu_sum": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulgroupnormleakyrelusum"
  },
  "gemm_sigmoid_scaling_residual_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmsigmoidscalingresidualadd"
  },
  "matmul_dropout_mean_softmax": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmuldropoutsoftmax"
  },
  "conv_transpose3d_softmax_sigmoid": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dsoftmaxsigmoid"
  },
  "conv3d_max_log_sum_exp_relu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dmaxlogsumexprelu"
  },
  "matmul_subtract_multiply_relu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulsubtractmultiplyrelu"
  },
  "conv_transpose2d_max_pool_hardtanh_mean_tanh": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dmaxpoolhardtanhmeantanh"
  },
  "conv3d_softmax_max_pool_max_pool": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dsoftmaxmaxpoolmaxpool"
  },
  "conv3d_scaling_tanh_multiply_sigmoid": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dscalingtanhmultiplysigmoid"
  },
  "conv_transpose2d_gelu_group_norm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dgelugroupnorm"
  },
  "conv_transpose2d_subtract_tanh": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dsubtracttanh"
  },
  "conv3d_leaky_relu_sum_clamp_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dleakyrelusumclampgelu"
  },
  "gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmlogsumexpleakyreluleakyrelugelugelu"
  },
  "conv2d_subtract_subtract_mish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dsubtractsubtractmish"
  },
  "conv2d_tanh_scaling_bias_add_max": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dtanhscalingbiasaddmax"
  },
  "conv2d_divide_leaky_relu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2ddivideleakyrelu"
  },
  "gemm_add_relu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmaddrelu"
  },
  "gemm_divide_sum_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmdividesumscaling"
  },
  "matmul_divide_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmuldividegelu"
  },
  "convtranspose2d_batchnorm_tanh_maxpool_groupnorm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dbatchnormtanhmaxpoolgroupnorm"
  },
  "conv_transpose2d_bias_add_clamp_scaling_clamp_divide": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dbiasaddclampscalingclampdivide"
  },
  "conv2d_multiply_leaky_relu_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dmultiplyleakyrelugelu"
  },
  "conv3d_relu_leaky_relu_gelu_sigmoid_bias_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dreluleakyrelugelusigmoidbiasadd"
  },
  "conv3d_mish_tanh": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dmishtanh"
  },
  "conv2d_add_scale_sigmoid_group_norm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2daddscalesigmoidgroupnorm"
  },
  "matmul_swish_sum_group_norm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulswishsumgroupnorm"
  },
  "matmul_sigmoid_sum": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulsigmoidsum"
  },
  "matmul_batch_norm_bias_add_divide_swish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulbatchnormbiasadddivideswish"
  },
  "matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulsummaxavgpoollogsumexplogsumex"
  },
  "conv2d_min_add_multiply": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dminaddmultipl"
  },
  "matmul_mish_mish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulmishmish"
  },
  "gemm_scaling_hard_tanh_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmscalinghardtanhgelu"
  },
  "conv_transpose3d_sum_residual_add_multiply_residual_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dsumresidualaddmultiplyresidualadd"
  },
  "conv2d_instance_norm_divide": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dinstancenormdivide"
  },
  "conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dgroupnormtanhhardswishresidualaddlogsumex"
  },
  "conv3d_min_softmax": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dminsoftmax"
  },
  "conv2d_batch_norm_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dbatchnormscaling"
  },
  "conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dlogsumexphardswishsubtractclam"
  },
  "conv_transpose2d_min_sum_gelu_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dminsumgeluadd"
  },
  "conv_transpose3d_layer_norm_gelu_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dlayernormgeluscaling"
  },
  "gemm_scale_batch_norm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmscalebatchnorm"
  },
  "gemm_group_norm_swish_multiply_swish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmgroupnormswishmultiplyswish"
  },
  "conv_transpose3d_avg_pool_clamp_softmax_multiply": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3davgpoolclampsoftmaxmultipl"
  },
  "matmul_min_subtract": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulminsubtract"
  },
  "conv3d_group_norm_mean": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dgroupnormmean"
  },
  "conv2d_avg_pool_sigmoid_sum": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2davgpoolsigmoidsum"
  },
  "gemm_relu_divide": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmreludivide"
  },
  "conv2d_mish_mish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dmishmish"
  },
  "gemm_scale_batchnorm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmscalebatchnorm"
  },
  "matmul_add_swish_tanh_gelu_hardtanh": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmuladdswishtanhgeluhardtanh"
  },
  "conv_transpose3d_leaky_relu_multiply_leaky_relu_max": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dleakyrelumultiplyleakyrelumax"
  },
  "conv3d_hardswish_relu_softmax_mean": {
    "category": "fuse",
    "level": "level2"
  },
  "gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmsubtractglobalavgpoollogsumexpgeluresidualadd"
  },
  "conv_transpose2d_mish_add_hardtanh_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dmishaddhardtanhscaling"
  },
  "conv_transpose3d_clamp_min_divide": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dclampmindivide"
  },
  "conv_transpose3d_multiply_max_global_avg_pool_clamp": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dmultiplymaxglobalavgpoolclam"
  },
  "conv3d_group_norm_min_clamp_dropout": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dgroupnormminclampdropout"
  },
  "conv2d_hard_swish_relu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dhardswishrelu"
  },
  "conv_transpose3d_max_pool_softmax_subtract_swish_max": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dmaxpoolsoftmaxsubtractswishmax"
  },
  "conv2d_gelu_global_avg_pool": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dgeluglobalavgpool"
  },
  "gemm_sigmoid_sum_log_sum_exp": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmsigmoidlogsumex"
  },
  "convtranspose3d_mean_add_softmax_tanh_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dmeanaddsoftmaxtanhscaling"
  },
  "conv2d_scaling_min": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dscalingmin"
  },
  "conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dmultiplyglobalavgpoolglobalavgpoolmean"
  },
  "conv2d_relu_hard_swish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dreluhardswish"
  },
  "conv2d_relu_bias_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2drelubiasadd"
  },
  "conv_transpose3d_scale_batch_norm_global_avg_pool": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dscalebatchnormglobalavgpool"
  },
  "gemm_bias_add_hardtanh_mish_group_norm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmbiasaddhardtanhmishgroupnorm"
  },
  "conv2d_subtract_hard_swish_max_pool_mish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dsubtracthardswishmaxpoolmish"
  },
  "matmul_avg_pool_gelu_scale_max": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulavgpoolgeluscalemax"
  },
  "matmul_scaling_residual_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulscalingresidualadd"
  },
  "selu": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "selu"
  },
  "swish": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "swish"
  },
  "leaky_relu": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "leakyrelu"
  },
  "min_gpt_new_gelu": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "mingptnewgelu"
  },
  "tanh": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "tanh"
  },
  "softplus": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "softplus"
  },
  "softsign": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "softsign"
  },
  "log_softmax": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "logsoftmax"
  },
  "hardsigmoid": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "hardsigmoid"
  },
  "softmax": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "softmax"
  },
  "relu": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "relu"
  },
  "hardtanh": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "hardtanh"
  },
  "sigmoid": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "sigmoid"
  },
  "elu": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "elu"
  },
  "gelu": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "gelu"
  },
  "conv_transposed_2d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dsquareinputsquarekernel"
  },
  "conv_transposed_3d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dsquareinputasymmetrickernel"
  },
  "conv_transposed_3d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dasymmetricinputsquarekernel"
  },
  "conv_depthwise_2d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convdepthwise2dasymmetricinputasymmetrickernel"
  },
  "conv_depthwise_2d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convdepthwise2dsquareinputasymmetrickernel"
  },
  "conv_standard_2d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard2dasymmetricinputsquarekernel"
  },
  "conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dasymmetricinputasymmetrickernelstridedpaddedgrouped"
  },
  "conv_depthwise_2d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convdepthwise2dsquareinputsquarekernel"
  },
  "conv_transposed_2d_asymmetric_input_square_kernel_dilated_padded_strided": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dasymmetricinputsquarekerneldilatedpaddedstrided"
  },
  "conv_standard_1d": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard1d"
  },
  "conv_standard_3d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard3dsquareinputasymmetrickernel"
  },
  "conv_standard_2d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard2dasymmetricinputasymmetrickernel"
  },
  "conv_transposed_3d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dsquareinputsquarekernel"
  },
  "conv_transposed_1d_dilated": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed1ddilated"
  },
  "conv_transposed_2d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dsquareinputasymmetrickernel"
  },
  "conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dasymmetricinputasymmetrickernelstridedgroupedpaddeddilated"
  },
  "conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed1dasymmetricinputsquarekernelpaddedstrideddilated"
  },
  "conv_transposed_2d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dasymmetricinputsquarekernel"
  },
  "conv_transposed_3d_square_input_square_kernel_padded_dilated_strided": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dsquareinputsquarekernelpaddeddilatedstrided"
  },
  "conv_depthwise_separable_2d": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convdepthwiseseparable2d"
  },
  "conv_standard_2d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard2dsquareinputasymmetrickernel"
  },
  "conv_standard_2d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard2dsquareinputsquarekernel"
  },
  "conv_depthwise_2d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convdepthwise2dasymmetricinputsquarekernel"
  },
  "conv_pointwise_2d": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convpointwise2d"
  },
  "conv_transposed_2d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dasymmetricinputasymmetrickernel"
  },
  "conv_standard_3d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard3dasymmetricinputsquarekernel"
  },
  "conv_transposed_3d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dasymmetricinputasymmetrickernel"
  },
  "conv_standard_3d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard3dsquareinputsquarekernel"
  },
  "conv_standard_1d_dilated_strided": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard1ddilatedstrided"
  },
  "conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dasymmetricinputasymmetrickernelpadded"
  },
  "conv_standard_3d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard3dasymmetricinputasymmetrickernel"
  },
  "conv_transposed_3d_asymmetric_input_square_kernel_strided_padded_grouped": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dasymmetricinputsquarekernelstridedpaddedgrouped"
  },
  "conv_transposed_1d": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed1d"
  },
  "conv_standard_2d_square_input_asymmetric_kernel_dilated_padded": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard2dsquareinputasymmetrickerneldilatedpadded"
  },
  "sgd": {
    "category": "optimizer",
    "level": "level1"
  },
  "rmsprop": {
    "category": "optimizer",
    "level": "level1"
  },
  "lamb": {
    "category": "optimizer",
    "level": "level1"
  },
  "adam": {
    "category": "optimizer",
    "level": "level1"
  },
  "adagrad": {
    "category": "optimizer",
    "level": "level1"
  },
  "sum_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1",
    "matched_ref": "sumreductionoveradimension"
  },
  "mean_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1",
    "matched_ref": "meanreductionoveradimension"
  },
  "product_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1",
    "matched_ref": "sumreductionoveradimension"
  },
  "max_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1",
    "matched_ref": "maxreductionoveradimension"
  },
  "min_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1",
    "matched_ref": "minreductionoveradimension"
  },
  "hinge_loss": {
    "category": "loss",
    "level": "level1",
    "matched_ref": "hingeloss"
  },
  "cross_entropy_loss": {
    "category": "loss",
    "level": "level1",
    "matched_ref": "crossentropyloss"
  },
  "triplet_margin_loss": {
    "category": "loss",
    "level": "level1",
    "matched_ref": "tripletmarginloss"
  },
  "mse_loss": {
    "category": "loss",
    "level": "level1",
    "matched_ref": "mseloss"
  },
  "kl_div_loss": {
    "category": "loss",
    "level": "level1",
    "matched_ref": "kldivloss"
  },
  "cosine_similarity_loss": {
    "category": "loss",
    "level": "level1"
  },
  "huber_loss": {
    "category": "loss",
    "level": "level1",
    "matched_ref": "huberloss"
  },
  "cumprod": {
    "category": "math",
    "level": "level1",
    "matched_ref": "cumprod"
  },
  "cumsum_exclusive": {
    "category": "math",
    "level": "level1",
    "matched_ref": "cumsumexclusive"
  },
  "matrix_scalar_multiplication": {
    "category": "math",
    "level": "level1",
    "matched_ref": "matrixscalarmultiplication"
  },
  "cumsum_reverse": {
    "category": "math",
    "level": "level1",
    "matched_ref": "cumsumreverse"
  },
  "cumsum": {
    "category": "math",
    "level": "level1",
    "matched_ref": "cumsum"
  },
  "masked_cumsum": {
    "category": "math",
    "level": "level1",
    "matched_ref": "maskedcumsum"
  },
  "division_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "power_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "elmentwise_mul_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "subtract_with_bias_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "add_bias_four_dim_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "logic_and_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "where_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "add_bias_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "max_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "clamp_broadcast": {
    "category": "broadcast",
    "level": "level1"
  },
  "squeeze_net_fire_module": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "squeezenetfiremodule"
  },
  "densenet121_transition_layer": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "densenet121transitionlayer"
  },
  "net_vlad_no_ghost_clusters": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "netvladnoghostclusters"
  },
  "lenet5": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "lenet5"
  },
  "efficientnet_b0": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "efficientnetb0"
  },
  "ltsm_cn": {
    "category": "arch",
    "level": "level3"
  },
  "squeeze_net": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "squeezenet"
  },
  "ltsm": {
    "category": "arch",
    "level": "level3"
  },
  "shufflenet": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "shufflenet"
  },
  "mobilenet_v1": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "mobilenetv1"
  },
  "shufflenet_unit": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "shufflenetunit"
  },
  "ltsm_hn": {
    "category": "arch",
    "level": "level3"
  },
  "densenet121_dense_block": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "densenet121denseblock"
  },
  "mamba_return_y": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "mamba2returny"
  },
  "vanilla_rnn": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "vanillarnn"
  },
  "swintransformer_v2": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "swintransformerv2"
  },
  "resnet101": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "resnet101"
  },
  "gru": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "gru"
  },
  "mini_gpt_block": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "minigptblock"
  },
  "vision_transformer": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "visiontransformer"
  },
  "ltsm_bidirectional": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "lstmbidirectional"
  },
  "vgg19": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "vgg19"
  },
  "min_gpt_causal_attention": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "mingptcausalattention"
  },
  "deep_narrow_mlp": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "deepnarrowmlp"
  },
  "shallow_wide_mlp": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "shallowwidemlp"
  },
  "unet_softmax": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "unetsoftmax"
  },
  "vgg16": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "vgg16"
  },
  "gru_bidirectional_hidden": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "grubidirectionalhidden"
  },
  "resnet18": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "resnet18"
  },
  "alexnet": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "alexnet"
  },
  "relu_self_attention": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "reluselfattention"
  },
  "regnet": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "regnet"
  },
  "googlenet_inception_module": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "googlenetinceptionmodule"
  },
  "vision_attention": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "visionattention"
  },
  "mlp": {
    "category": "arch",
    "level": "level3"
  },
  "densenet121": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "densenet121"
  },
  "swin_mlp": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "swinmlp"
  },
  "googlenet_inception_v1": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "googlenetinceptionv1"
  },
  "efficientnet_b2": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "efficientnetb2"
  },
  "resnet_basic_block": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "resnetbasicblock"
  },
  "densenet201": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "densenet201"
  },
  "mobilenet_v2": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "mobilenetv2"
  },
  "gru_hidden": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "gruhidden"
  },
  "vanilla_rnn_hidden": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "vanillarnnhidden"
  },
  "net_vlad_with_ghost_clusters": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "netvladwithghostclusters"
  },
  "convolutional_vision_transformer": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "convolutionalvisiontransformer"
  },
  "mamba_return_final_state": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "mamba2returnfinalstate"
  },
  "efficientnet_mb_conv": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "efficientnetmbconv"
  },
  "gru_birectional": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "grubidirectional"
  },
  "efficientnet_b1": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "efficientnetb1"
  },
  "layer_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "layernorm"
  },
  "l1_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "l1norm"
  },
  "l2_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "l2norm"
  },
  "group_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "groupnorm"
  },
  "frobenius_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "frobeniusnorm"
  },
  "instance_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "instancenorm"
  },
  "rms_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "rmsnorm"
  },
  "batch_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "batchnorm"
  }
}


kernel_bench_level = {
  "upsample_grid_sample": {
    "category": "resize",
    "level": "unknown"
  },
  "trilinear_upsample": {
    "category": "resize",
    "level": "unknown"
  },
  "grid_sample_affine": {
    "category": "resize",
    "level": "unknown"
  },
  "bicubic_upsample": {
    "category": "resize",
    "level": "unknown"
  },
  "grid_sample_random_warp": {
    "category": "resize",
    "level": "unknown"
  },
  "downsample_bilinear": {
    "category": "resize",
    "level": "unknown"
  },
  "bilinear_upsample": {
    "category": "resize",
    "level": "unknown"
  },
  "nearest_neighbor_upsample": {
    "category": "resize",
    "level": "unknown"
  },
  "interpolate_dynamic": {
    "category": "resize",
    "level": "unknown"
  },
  "resize_with_antialias": {
    "category": "resize",
    "level": "unknown"
  },
  "average_pooling_1d": {
    "category": "pooling",
    "level": "level1",
    "matched_ref": "averagepooling1d"
  },
  "average_pooling_2d": {
    "category": "pooling",
    "level": "level1",
    "matched_ref": "averagepooling2d"
  },
  "average_pooling_3d": {
    "category": "pooling",
    "level": "level1",
    "matched_ref": "averagepooling3d"
  },
  "max_pooling_3d": {
    "category": "pooling",
    "level": "level1",
    "matched_ref": "maxpooling3d"
  },
  "max_pooling_2d": {
    "category": "pooling",
    "level": "level1",
    "matched_ref": "maxpooling2d"
  },
  "max_pooling_1d": {
    "category": "pooling",
    "level": "level1",
    "matched_ref": "maxpooling1d"
  },
  "batched_matrix_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "batchedmatrixmultiplication"
  },
  "matmul_with_diagonal_matrices": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithdiagonalmatrices"
  },
  "four_dim_tensor_matrix_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "3dtensormatrixmultiplication"
  },
  "matmul_with_small_k_dimension": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithsmallkdimension"
  },
  "matmul_for_symmetric_matrices": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulforsymmetricmatrices"
  },
  "matmul_with_large_k_dimension": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithlargekdimension"
  },
  "matmul_with_transposed_b": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithtransposedb"
  },
  "matmul_with_transposed_a": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithtransposeda"
  },
  "matmul_for_lower_triangular_matrices": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulforlowertriangularmatrices"
  },
  "three_dim_tensor_matrix_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "3dtensormatrixmultiplication"
  },
  "square_matrix_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "squarematrixmultiplication"
  },
  "matmul_with_transposed_both": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithtransposedboth"
  },
  "matmul_with_irregular_shapes": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulwithirregularshapes"
  },
  "matmul_for_upper_triangular_matrices": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matmulforuppertriangularmatrices"
  },
  "tall_skinny_matrix_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "tallskinnymatrixmultiplication"
  },
  "standard_matrix_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "standardmatrixmultiplication"
  },
  "matrix_vector_multiplication": {
    "category": "matmul",
    "level": "level1",
    "matched_ref": "matrixvectormultiplication"
  },
  "index_copy": {
    "category": "index",
    "level": "unknown"
  },
  "scatter_add": {
    "category": "index",
    "level": "unknown"
  },
  "gather": {
    "category": "index",
    "level": "unknown"
  },
  "index_select": {
    "category": "index",
    "level": "unknown"
  },
  "masked_fill": {
    "category": "index",
    "level": "unknown"
  },
  "index_add": {
    "category": "index",
    "level": "unknown"
  },
  "scatter": {
    "category": "index",
    "level": "unknown"
  },
  "argmax_over_a_dimension": {
    "category": "index",
    "level": "level1",
    "matched_ref": "argmaxoveradimension"
  },
  "take_along_dim": {
    "category": "index",
    "level": "unknown"
  },
  "inplace_update": {
    "category": "index",
    "level": "unknown"
  },
  "embedding": {
    "category": "index",
    "level": "unknown"
  },
  "argmin_over_a_dimension": {
    "category": "index",
    "level": "level1",
    "matched_ref": "argminoveradimension"
  },
  "matmul_gelu_softmax": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulgelusoftmax"
  },
  "conv3d_multiply_instance_norm_clamp_multiply_max": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dmultiplyinstancenormclampmultiplymax"
  },
  "conv2d_subtract_tanh_subtract_avg_pool": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dsubtracttanhsubtractavgpool"
  },
  "conv3d_divide_max_global_avg_pool_bias_add_sum": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3ddividemaxglobalavgpoolbiasaddsum"
  },
  "conv2d_group_norm_scale_max_pool_clamp": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dgroupnormscalemaxpoolclam"
  },
  "conv2d_activation_batch_norm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dactivationbatchnorm"
  },
  "conv_transpose3d_sum_layer_norm_avg_pool_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dsumlayernormavgpoolgelu"
  },
  "gemm_group_norm_min_bias_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmgroupnormminbiasadd"
  },
  "gemm_batch_norm_scaling_softmax": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmbatchnormscalingsoftmax"
  },
  "conv_transpose3d_add_hard_swish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3daddhardswish"
  },
  "conv_transpose3d_max_max_sum": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dmaxmaxsum"
  },
  "convtranspose3d_relu_groupnorm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3drelugroupnorm"
  },
  "matmul_scale_residual_add_clamp_log_sum_exp_mish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulscaleresidualaddclamplogsumexpmish"
  },
  "convtranspose2d_softmax_biasadd_scaling_sigmoid": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dsoftmaxbiasaddscalingsigmoid"
  },
  "bmm_instance_norm_sum_residual_add_multiply": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "bmminstancenormsumresidualaddmultipl"
  },
  "conv_transpose3d_scaling_avg_pool_bias_add_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dscalingavgpoolbiasaddscaling"
  },
  "gemm_group_norm_hardtanh": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmgroupnormhardtanh"
  },
  "matmul_max_pool_sum_scale": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulmaxpoolsumscale"
  },
  "conv_transpose3d_batch_norm_subtract": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dbatchnormsubtract"
  },
  "gemm_batch_norm_gelu_group_norm_mean_relu": {
    "category": "fuse",
    "level": "unknown"
  },
  "conv2d_min_tanh_tanh": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dmintanhtanh"
  },
  "conv_transpose3d_swish_group_norm_hard_swish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dswishgroupnormhardswish"
  },
  "gemm_swish_divide_clamp_tanh_clamp": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmswishdivideclamptanhclam"
  },
  "convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dglobalavgpoolbiasaddlogsumexpsummultipl"
  },
  "conv_transpose3d_batch_norm_avg_pool_avg_pool": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dbatchnormavgpoolavgpool"
  },
  "matmul_swish_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulswishscaling"
  },
  "gemm_max_subtract_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmmaxsubtractgelu"
  },
  "gemm_multiply_leakyrelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmmultiplyleakyrelu"
  },
  "conv_transpose2d_add_min_gelu_multiply": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2daddmingelumultipl"
  },
  "matmul_group_norm_leaky_relu_sum": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulgroupnormleakyrelusum"
  },
  "gemm_sigmoid_scaling_residual_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmsigmoidscalingresidualadd"
  },
  "matmul_dropout_mean_softmax": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmuldropoutsoftmax"
  },
  "conv_transpose3d_softmax_sigmoid": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dsoftmaxsigmoid"
  },
  "conv3d_max_log_sum_exp_relu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dmaxlogsumexprelu"
  },
  "matmul_subtract_multiply_relu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulsubtractmultiplyrelu"
  },
  "conv_transpose2d_max_pool_hardtanh_mean_tanh": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dmaxpoolhardtanhmeantanh"
  },
  "conv3d_softmax_max_pool_max_pool": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dsoftmaxmaxpoolmaxpool"
  },
  "conv3d_scaling_tanh_multiply_sigmoid": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dscalingtanhmultiplysigmoid"
  },
  "conv_transpose2d_gelu_group_norm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dgelugroupnorm"
  },
  "conv_transpose2d_subtract_tanh": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dsubtracttanh"
  },
  "conv3d_leaky_relu_sum_clamp_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dleakyrelusumclampgelu"
  },
  "gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmlogsumexpleakyreluleakyrelugelugelu"
  },
  "conv2d_subtract_subtract_mish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dsubtractsubtractmish"
  },
  "conv2d_tanh_scaling_bias_add_max": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dtanhscalingbiasaddmax"
  },
  "conv2d_divide_leaky_relu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2ddivideleakyrelu"
  },
  "gemm_add_relu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmaddrelu"
  },
  "gemm_divide_sum_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmdividesumscaling"
  },
  "matmul_divide_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmuldividegelu"
  },
  "convtranspose2d_batchnorm_tanh_maxpool_groupnorm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dbatchnormtanhmaxpoolgroupnorm"
  },
  "conv_transpose2d_bias_add_clamp_scaling_clamp_divide": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dbiasaddclampscalingclampdivide"
  },
  "conv2d_multiply_leaky_relu_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dmultiplyleakyrelugelu"
  },
  "conv3d_relu_leaky_relu_gelu_sigmoid_bias_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dreluleakyrelugelusigmoidbiasadd"
  },
  "conv3d_mish_tanh": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dmishtanh"
  },
  "conv2d_add_scale_sigmoid_group_norm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2daddscalesigmoidgroupnorm"
  },
  "matmul_swish_sum_group_norm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulswishsumgroupnorm"
  },
  "matmul_sigmoid_sum": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulsigmoidsum"
  },
  "matmul_batch_norm_bias_add_divide_swish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulbatchnormbiasadddivideswish"
  },
  "matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulsummaxavgpoollogsumexplogsumex"
  },
  "conv2d_min_add_multiply": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dminaddmultipl"
  },
  "matmul_mish_mish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulmishmish"
  },
  "gemm_scaling_hard_tanh_gelu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmscalinghardtanhgelu"
  },
  "conv_transpose3d_sum_residual_add_multiply_residual_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dsumresidualaddmultiplyresidualadd"
  },
  "conv2d_instance_norm_divide": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dinstancenormdivide"
  },
  "conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dgroupnormtanhhardswishresidualaddlogsumex"
  },
  "conv3d_min_softmax": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dminsoftmax"
  },
  "conv2d_batch_norm_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dbatchnormscaling"
  },
  "conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dlogsumexphardswishsubtractclam"
  },
  "conv_transpose2d_min_sum_gelu_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dminsumgeluadd"
  },
  "conv_transpose3d_layer_norm_gelu_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dlayernormgeluscaling"
  },
  "gemm_scale_batch_norm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmscalebatchnorm"
  },
  "gemm_group_norm_swish_multiply_swish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmgroupnormswishmultiplyswish"
  },
  "conv_transpose3d_avg_pool_clamp_softmax_multiply": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3davgpoolclampsoftmaxmultipl"
  },
  "matmul_min_subtract": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulminsubtract"
  },
  "conv3d_group_norm_mean": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dgroupnormmean"
  },
  "conv2d_avg_pool_sigmoid_sum": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2davgpoolsigmoidsum"
  },
  "gemm_relu_divide": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmreludivide"
  },
  "conv2d_mish_mish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dmishmish"
  },
  "gemm_scale_batchnorm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmscalebatchnorm"
  },
  "matmul_add_swish_tanh_gelu_hardtanh": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmuladdswishtanhgeluhardtanh"
  },
  "conv_transpose3d_leaky_relu_multiply_leaky_relu_max": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dleakyrelumultiplyleakyrelumax"
  },
  "conv3d_hardswish_relu_softmax_mean": {
    "category": "fuse",
    "level": "unknown"
  },
  "gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmsubtractglobalavgpoollogsumexpgeluresidualadd"
  },
  "conv_transpose2d_mish_add_hardtanh_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dmishaddhardtanhscaling"
  },
  "conv_transpose3d_clamp_min_divide": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dclampmindivide"
  },
  "conv_transpose3d_multiply_max_global_avg_pool_clamp": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dmultiplymaxglobalavgpoolclam"
  },
  "conv3d_group_norm_min_clamp_dropout": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv3dgroupnormminclampdropout"
  },
  "conv2d_hard_swish_relu": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dhardswishrelu"
  },
  "conv_transpose3d_max_pool_softmax_subtract_swish_max": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dmaxpoolsoftmaxsubtractswishmax"
  },
  "conv2d_gelu_global_avg_pool": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dgeluglobalavgpool"
  },
  "gemm_sigmoid_sum_log_sum_exp": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmsigmoidlogsumex"
  },
  "convtranspose3d_mean_add_softmax_tanh_scaling": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dmeanaddsoftmaxtanhscaling"
  },
  "conv2d_scaling_min": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dscalingmin"
  },
  "conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose2dmultiplyglobalavgpoolglobalavgpoolmean"
  },
  "conv2d_relu_hard_swish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dreluhardswish"
  },
  "conv2d_relu_bias_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2drelubiasadd"
  },
  "conv_transpose3d_scale_batch_norm_global_avg_pool": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "convtranspose3dscalebatchnormglobalavgpool"
  },
  "gemm_bias_add_hardtanh_mish_group_norm": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "gemmbiasaddhardtanhmishgroupnorm"
  },
  "conv2d_subtract_hard_swish_max_pool_mish": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "conv2dsubtracthardswishmaxpoolmish"
  },
  "matmul_avg_pool_gelu_scale_max": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulavgpoolgeluscalemax"
  },
  "matmul_scaling_residual_add": {
    "category": "fuse",
    "level": "level2",
    "matched_ref": "matmulscalingresidualadd"
  },
  "selu": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "selu"
  },
  "swish": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "swish"
  },
  "leaky_relu": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "leakyrelu"
  },
  "min_gpt_new_gelu": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "mingptnewgelu"
  },
  "tanh": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "tanh"
  },
  "softplus": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "softplus"
  },
  "softsign": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "softsign"
  },
  "log_softmax": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "logsoftmax"
  },
  "hardsigmoid": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "hardsigmoid"
  },
  "softmax": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "softmax"
  },
  "relu": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "relu"
  },
  "hardtanh": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "hardtanh"
  },
  "sigmoid": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "sigmoid"
  },
  "elu": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "elu"
  },
  "gelu": {
    "category": "activation",
    "level": "level1",
    "matched_ref": "gelu"
  },
  "conv_transposed_2d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dsquareinputsquarekernel"
  },
  "conv_transposed_3d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dsquareinputasymmetrickernel"
  },
  "conv_transposed_3d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dasymmetricinputsquarekernel"
  },
  "conv_depthwise_2d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convdepthwise2dasymmetricinputasymmetrickernel"
  },
  "conv_depthwise_2d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convdepthwise2dsquareinputasymmetrickernel"
  },
  "conv_standard_2d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard2dasymmetricinputsquarekernel"
  },
  "conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dasymmetricinputasymmetrickernelstridedpaddedgrouped"
  },
  "conv_depthwise_2d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convdepthwise2dsquareinputsquarekernel"
  },
  "conv_transposed_2d_asymmetric_input_square_kernel_dilated_padded_strided": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dasymmetricinputsquarekerneldilatedpaddedstrided"
  },
  "conv_standard_1d": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard1d"
  },
  "conv_standard_3d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard3dsquareinputasymmetrickernel"
  },
  "conv_standard_2d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard2dasymmetricinputasymmetrickernel"
  },
  "conv_transposed_3d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dsquareinputsquarekernel"
  },
  "conv_transposed_1d_dilated": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed1ddilated"
  },
  "conv_transposed_2d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dsquareinputasymmetrickernel"
  },
  "conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dasymmetricinputasymmetrickernelstridedgroupedpaddeddilated"
  },
  "conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed1dasymmetricinputsquarekernelpaddedstrideddilated"
  },
  "conv_transposed_2d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dasymmetricinputsquarekernel"
  },
  "conv_transposed_3d_square_input_square_kernel_padded_dilated_strided": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dsquareinputsquarekernelpaddeddilatedstrided"
  },
  "conv_depthwise_separable_2d": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convdepthwiseseparable2d"
  },
  "conv_standard_2d_square_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard2dsquareinputasymmetrickernel"
  },
  "conv_standard_2d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard2dsquareinputsquarekernel"
  },
  "conv_depthwise_2d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convdepthwise2dasymmetricinputsquarekernel"
  },
  "conv_pointwise_2d": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convpointwise2d"
  },
  "conv_transposed_2d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dasymmetricinputasymmetrickernel"
  },
  "conv_standard_3d_asymmetric_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard3dasymmetricinputsquarekernel"
  },
  "conv_transposed_3d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dasymmetricinputasymmetrickernel"
  },
  "conv_standard_3d_square_input_square_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard3dsquareinputsquarekernel"
  },
  "conv_standard_1d_dilated_strided": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard1ddilatedstrided"
  },
  "conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed2dasymmetricinputasymmetrickernelpadded"
  },
  "conv_standard_3d_asymmetric_input_asymmetric_kernel": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard3dasymmetricinputasymmetrickernel"
  },
  "conv_transposed_3d_asymmetric_input_square_kernel_strided_padded_grouped": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed3dasymmetricinputsquarekernelstridedpaddedgrouped"
  },
  "conv_transposed_1d": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convtransposed1d"
  },
  "conv_standard_2d_square_input_asymmetric_kernel_dilated_padded": {
    "category": "convolution",
    "level": "level1",
    "matched_ref": "convstandard2dsquareinputasymmetrickerneldilatedpadded"
  },
  "sgd": {
    "category": "optimizer",
    "level": "unknown"
  },
  "rmsprop": {
    "category": "optimizer",
    "level": "unknown"
  },
  "lamb": {
    "category": "optimizer",
    "level": "unknown"
  },
  "adam": {
    "category": "optimizer",
    "level": "unknown"
  },
  "adagrad": {
    "category": "optimizer",
    "level": "unknown"
  },
  "sum_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1",
    "matched_ref": "sumreductionoveradimension"
  },
  "mean_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1",
    "matched_ref": "meanreductionoveradimension"
  },
  "product_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1",
    "matched_ref": "sumreductionoveradimension"
  },
  "max_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1",
    "matched_ref": "maxreductionoveradimension"
  },
  "min_reduction_over_a_dimension": {
    "category": "reduce",
    "level": "level1",
    "matched_ref": "minreductionoveradimension"
  },
  "hinge_loss": {
    "category": "loss",
    "level": "level1",
    "matched_ref": "hingeloss"
  },
  "cross_entropy_loss": {
    "category": "loss",
    "level": "level1",
    "matched_ref": "crossentropyloss"
  },
  "triplet_margin_loss": {
    "category": "loss",
    "level": "level1",
    "matched_ref": "tripletmarginloss"
  },
  "mse_loss": {
    "category": "loss",
    "level": "level1",
    "matched_ref": "mseloss"
  },
  "kl_div_loss": {
    "category": "loss",
    "level": "level1",
    "matched_ref": "kldivloss"
  },
  "cosine_similarity_loss": {
    "category": "loss",
    "level": "unknown"
  },
  "huber_loss": {
    "category": "loss",
    "level": "level1",
    "matched_ref": "huberloss"
  },
  "cumprod": {
    "category": "math",
    "level": "level1",
    "matched_ref": "cumprod"
  },
  "cumsum_exclusive": {
    "category": "math",
    "level": "level1",
    "matched_ref": "cumsumexclusive"
  },
  "matrix_scalar_multiplication": {
    "category": "math",
    "level": "level1",
    "matched_ref": "matrixscalarmultiplication"
  },
  "cumsum_reverse": {
    "category": "math",
    "level": "level1",
    "matched_ref": "cumsumreverse"
  },
  "cumsum": {
    "category": "math",
    "level": "level1",
    "matched_ref": "cumsum"
  },
  "masked_cumsum": {
    "category": "math",
    "level": "level1",
    "matched_ref": "maskedcumsum"
  },
  "division_broadcast": {
    "category": "broadcast",
    "level": "unknown"
  },
  "power_broadcast": {
    "category": "broadcast",
    "level": "unknown"
  },
  "elmentwise_mul_broadcast": {
    "category": "broadcast",
    "level": "unknown"
  },
  "subtract_with_bias_broadcast": {
    "category": "broadcast",
    "level": "unknown"
  },
  "add_bias_four_dim_broadcast": {
    "category": "broadcast",
    "level": "unknown"
  },
  "logic_and_broadcast": {
    "category": "broadcast",
    "level": "unknown"
  },
  "where_broadcast": {
    "category": "broadcast",
    "level": "unknown"
  },
  "add_bias_broadcast": {
    "category": "broadcast",
    "level": "unknown"
  },
  "max_broadcast": {
    "category": "broadcast",
    "level": "unknown"
  },
  "clamp_broadcast": {
    "category": "broadcast",
    "level": "unknown"
  },
  "squeeze_net_fire_module": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "squeezenetfiremodule"
  },
  "densenet121_transition_layer": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "densenet121transitionlayer"
  },
  "net_vlad_no_ghost_clusters": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "netvladnoghostclusters"
  },
  "lenet5": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "lenet5"
  },
  "efficientnet_b0": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "efficientnetb0"
  },
  "ltsm_cn": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "lstmcn"
  },
  "squeeze_net": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "squeezenet"
  },
  "ltsm": {
    "category": "arch",
    "level": "unknown"
  },
  "shufflenet": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "shufflenet"
  },
  "mobilenet_v1": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "mobilenetv1"
  },
  "shufflenet_unit": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "shufflenetunit"
  },
  "ltsm_hn": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "lstmhn"
  },
  "densenet121_dense_block": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "densenet121denseblock"
  },
  "mamba_return_y": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "mamba2returny"
  },
  "vanilla_rnn": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "vanillarnn"
  },
  "swintransformer_v2": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "swintransformerv2"
  },
  "resnet101": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "resnet101"
  },
  "gru": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "gru"
  },
  "mini_gpt_block": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "minigptblock"
  },
  "vision_transformer": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "visiontransformer"
  },
  "ltsm_bidirectional": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "lstmbidirectional"
  },
  "vgg19": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "vgg19"
  },
  "min_gpt_causal_attention": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "mingptcausalattention"
  },
  "deep_narrow_mlp": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "deepnarrowmlp"
  },
  "shallow_wide_mlp": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "shallowwidemlp"
  },
  "unet_softmax": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "unetsoftmax"
  },
  "vgg16": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "vgg16"
  },
  "gru_bidirectional_hidden": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "grubidirectionalhidden"
  },
  "resnet18": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "resnet18"
  },
  "alexnet": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "alexnet"
  },
  "relu_self_attention": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "reluselfattention"
  },
  "regnet": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "regnet"
  },
  "googlenet_inception_module": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "googlenetinceptionmodule"
  },
  "vision_attention": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "visionattention"
  },
  "mlp": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "mlp"
  },
  "densenet121": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "densenet121"
  },
  "swin_mlp": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "swinmlp"
  },
  "googlenet_inception_v1": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "googlenetinceptionv1"
  },
  "efficientnet_b2": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "efficientnetb2"
  },
  "resnet_basic_block": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "resnetbasicblock"
  },
  "densenet201": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "densenet201"
  },
  "mobilenet_v2": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "mobilenetv2"
  },
  "gru_hidden": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "gruhidden"
  },
  "vanilla_rnn_hidden": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "vanillarnnhidden"
  },
  "net_vlad_with_ghost_clusters": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "netvladwithghostclusters"
  },
  "convolutional_vision_transformer": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "convolutionalvisiontransformer"
  },
  "mamba_return_final_state": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "mamba2returnfinalstate"
  },
  "efficientnet_mb_conv": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "efficientnetmbconv"
  },
  "gru_birectional": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "grubidirectional"
  },
  "efficientnet_b1": {
    "category": "arch",
    "level": "level3",
    "matched_ref": "efficientnetb1"
  },
  "layer_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "layernorm"
  },
  "l1_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "l1norm"
  },
  "l2_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "l2norm"
  },
  "group_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "groupnorm"
  },
  "frobenius_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "frobeniusnorm"
  },
  "instance_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "instancenorm"
  },
  "rms_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "rmsnorm"
  },
  "batch_norm": {
    "category": "normalization",
    "level": "level1",
    "matched_ref": "batchnorm"
  }
}