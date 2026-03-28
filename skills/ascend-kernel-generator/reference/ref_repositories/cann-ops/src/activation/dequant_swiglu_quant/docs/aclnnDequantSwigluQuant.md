# aclnnDequantSwigluQuant

## 支持的产品型号
- Atlas A2 训练系列产品。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnDequantSwigluQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDequantSwigluQuant”接口执行计算。

- `aclnnStatus aclnnDequantSwigluQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *weightScaleOptional, const aclTensor *activationScaleOptional, const aclTensor *biasOptional, const aclTensor *quantScaleOptional, const aclTensor *quantOffsetOptional, const aclTensor *groupIndexOptional, bool activateLeft, const char* quantMode, aclTensor *y, aclTensor *scale, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnDequantSwigluQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述
- 算子功能：在Swish门控线性单元激活函数前后添加dequant和quant操作，实现x的DequantSwigluQuant计算。  
- 计算公式：  

  $$
  dequantOut_i = Dequant(x_i)
  $$


  $$
  swigluOut_i = Swiglu(dequantOut_i)=Swish(A_i)*B_i
  $$


  $$
  out_i = Quant(swigluOut_i)
  $$

  其中，A<sub>i</sub>表示dequantOut<sub>i</sub>的前半部分，B<sub>i</sub>表示dequantOut<sub>i</sub>的后半部分。

## aclnnDequantSwigluQuantGetWorkspaceSize
- **参数说明**：
  - x（aclTensor*，计算输入）：输入待处理的数据，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、INT32。支持非连续的Tensor，数据格式支持ND。shape为(N..., H)，最后一维需要是2的倍数，且x的维数必须大于1维。
  - weightScaleOptional（aclTensor*，计算输入）：weight的反量化scale，Device侧的aclTensor，数据类型支持FLOAT。支持非连续的Tensor，数据格式支持ND。shape支持1维，shape表示为[H]，且取值H和x最后一维保持一致。可选参数，支持传空指针。
  - activationScaleOptional（aclTensor*，计算输入）：激活函数的反量化scale，Device侧的aclTensor，数据类型支持FLOAT。支持非连续的Tensor，数据格式支持ND。shape为[N..., 1]，最后一维为1，其余和x保持一致。可选参数，支持传空指针。
  - biasOptional（aclTensor*，计算输入）：Matmul的bias，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。支持非连续的Tensor，数据格式支持ND。shape支持1维，shape表示为[H]，且取值H和x最后一维保持一致。可选参数，支持传空指针。
  - quantScaleOptional（aclTensor*，计算输入）：量化的scale，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16。支持非连续的Tensor，数据格式支持ND。当quantMode为static时，shape为1维，值为1，shape表示为shape[1]；quantMode为dynamic时，shape维数为1维，值为x的最后一维的二分之一，shape表示为shape[H/2]。可选参数，支持传空指针。
  - quantOffsetOptional（aclTensor*，计算输入）：量化的offset，Device侧的aclTensor，数据类型支持FLOAT。支持非连续的Tensor，数据格式支持ND。当quantMode为static时，shape为1维，值为1，shape表示为shape[1]；quantMode为dynamic时，shape维数为1维，值为x的最后一维的二分之一，shape表示为shape[H/2]。可选参数，支持传空指针。
  - groupIndexOptional（aclTensor*，计算输入）：MoE分组需要的group_index，Device侧的aclTensor，数据类型支持INT32、INT64。支持非连续的Tensor，数据格式支持ND。shape支持1维Tensor。可选参数，支持传空指针。
  - activateLeft（bool，入参）：表示是否对输入的左半部分做swiglu激活，数据类型支持bool。当值为false时，对输入的右半部分做激活。
  - quantMode（char*，入参）：支持“dynamic”和“static"，表示使用动态量化还是静态量化，数据类型支持string。
  - y（aclTensor*，计算输出）：Device侧的aclTensor，数据类型支持INT8，支持非连续的Tensor，数据格式支持ND。
  - scale（aclTensor*，计算输出）：Device侧的aclTensor，数据类型支持FLOAT，支持非连续的Tensor，数据格式支持ND。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：
  aclnnStatus：返回状态码。

   ```
   第一段接口完成入参校验，出现以下场景时报错：
   返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的x或y是空指针。
   返回161002（ACLNN_ERR_PARAM_INVALID）：1. 输入或输出的数据类型不在支持的范围内。
                                         2. 输入或输出的参数维度不在支持的范围内。
                                         3. quantMode不在指定的取值范围内。 
   ```


## aclnnDequantSwigluQuant
- **参数说明**：
  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnDequantSwigluQuantGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：
  aclnnStatus：返回状态码。


## 约束与限制
- x的最后一维需要是2的倍数，且x的维数必须大于1维。
- 当quantMode为static时，quantScaleOptional和quantOffsetOptional为1维，值为1；quantMode为dynamic时，quantScaleOptional和quantOffsetOptional的维数为1维，值为x的最后一维除2。