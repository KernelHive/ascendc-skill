# aclnnConvolution

## 支持的产品型号
- 昇腾910B AI处理器。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnConvolutionGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnConvolution”接口执行计算。

* `aclnnStatus aclnnConvolutionGetWorkspaceSize( const aclTensor *input, const aclTensor *weight,  const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool transposed, const aclIntArray *outputPadding,  const int64_t groups, aclTensor *output, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnConvolution(void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

- 算子功能：实现卷积功能，支持3D卷积，同时支持空洞卷积、分组卷积。

- 计算公式：

  我们假定输入（input）的shape是 $(N, C_{\text{in}}, D_{\text{in}}，H, W)$ ，（weight）的shape是 $(C_{\text{out}}, C_{\text{in}}，K_{\text{d}}，K_h, K_w)$，输出（output）的shape是 $(N, C_{\text{out}}, D_{\text{out}}，H_{\text{out}}, W_{\text{out}})$，那输出将被表示为：

  $$
    \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
  $$

  其中，$\star$表示互相关的计算，根据卷积输入的dim，卷积的类型（空洞卷积、分组卷积）而定。$N$代表batch size，$C$代表通道数，$W$和$H$分别代表宽和高，相应输出维度的计算公式如下：

  $$
    H_{\text{out}}=[(H + 2 * padding[0] - dilation[0] * (K_h - 1) - 1 ) / stride[0]] + 1 \\
    W_{\text{out}}=[(W + 2 * padding[1] - dilation[1] * (K_w - 1) - 1 ) / stride[1]] + 1 \\
    D_{\text{out}}=[(D + 2 * padding[2] - dilation[2] * (K_d - 1) - 1 ) / stride[2]] + 1
  $$


## aclnnConvolutionGetWorkspaceSize

- **参数说明：**
  * input(aclTensor *, 计算输入)：公式中的`input`，数据类型支持FLOAT16，BFLOAT16，FLOAT16，支持非连续的Tensor，数据格式为NCDHW。数据类型需要与weight满足数据类型推导规则。
  * weight(aclTensor *, 计算输入)：公式中的`weight`，支持非连续的Tensor，数据格式为NCDHW。数据类型需要与input满足数据类型推导规则。
  * bias(aclTensor *, 计算输入)：公式中的`bias`，数据类型支持FLOAT16，BFLOAT16，FLOAT16，支持非连续的Tensor，数据格式为NCDHW, ND。
  * stride(aclIntArray *, 计算输入)：卷积扫描步长，数组长度需等于input的维度减2。其值应该大于0。
  * padding(aclIntArray *, 计算输入)：对input的填充，conv3d数组长度为3。其值应该大于等于0。
  * dilation(aclIntArray *, 计算输入)：卷积核中元素的间隔，数组长度需等于input的维度减2。其值应该大于0。
  * transposed(bool, 计算输入)：是否为转置卷积, 暂未支持，给FALSE即可。
  * outputPadding(aclIntArray *, 计算输入)：其值应大于等于0，且小于stride或dilation对应维度的值。非转置卷积情况下，忽略该属性配置。
  * groups(int64_t, 计算输入)：表示从输入通道到输出通道的块链接个数，数值必须大于0，且满足groups*weight的C维度=input的C维度。
  * cubeMathType(int8_t, 计算输入)：用于判断Cube单元应该使用哪种计算逻辑进行运算，数据类型为INT8，注意：如果输入的数据类型存在互推导关系，该参数默认对互推导后的数据类型进行处理。支持的枚举值如下：
    * 0:KEEP_DTYPE，保持输入的数据类型进行计算。
    * 1:ALLOW_FP32_DOWN_PRECISION，允许将输入数据降精度计算。
      - 昇腾910B AI处理器：当输入是FLOAT，允许转换为HFLOAT32计算。当输入为其他数据类型时不做处理。
    * 2:USE_FP16，允许转换为数据类型FLOAT16进行计算。
      - 昇腾910B AI处理器：当输入是BFLOAT16时不支持该选项。
    * 3:USE_HF32，允许转换为数据类型HFLOAT32计算。
      - 昇腾910B AI处理器：当输入是FLOAT，转换为HFLOAT32计算。当输入为其他数据类型时不做处理。
  * output(aclTensor *, 计算输出)：公式中的`out`，数据格式为NCDHW。数据类型需要与input与weight推导之后的数据类型保持一致。
    - 昇腾910B AI处理器：数据类型仅支持FLOAT，FLOAT16，BFLOAT16。
  * workspaceSize(uint64_t *, 出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor **, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus： 返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 ACLNN_ERR_PARAM_NULLPTR: 1. 传入的指针类型入参是空指针。
  161002 ACLNN_ERR_PARAM_INVALID: 1. input，weight，bias，output数据类型和数据格式不在支持的范围之内。
  2. stride, padding, dilation 输入shape不对。
  3. input和output数据类型不一致；卷积正向算子支持input和output数据类型不一致，不会触发该类型报错。
  4. groups为0或者大于0的情况下，weight和input通道数不满足要求。
  5. output的shape不满足infershape结果。
  6. input传入空tensor中任意维度为零的均不满足要求。
  7. input空间尺度在padding操作后小于weight(经过dilation扩张（如存在dilation>1的情况）)的空间尺度（非transpose模式下）。
  8. stride, dilation小于0情况下不满足要求。
  9. 当前处理器不支持卷积。
  561103 ACLNN_ERR_INNER_NULLPTR: 1. API内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。
  361001 ACLNN_ERR_RUNTIME_ERROR：1. API调用npu runtime的接口异常，如SocVersion不支持。
  ```


## aclnnConvolution

- **参数说明：**

* workspace(void*, 入参)：在Device侧申请的workspace内存地址。
* workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnConvolutionGetWorkspaceSize获取。
* executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
* stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

## 约束与限制

- 3D卷积仅支持transposed为false且输入数据类型为FLOAT16，BFLOAT16，FLOAT32场景。
- 由于硬件资源限制，算子在部分参数取值组合场景下会执行失败，请根据日志信息提示分析并排查问题。若无法解决，请单击[Link](https://www.hiascend.com/support)获取技术支持。