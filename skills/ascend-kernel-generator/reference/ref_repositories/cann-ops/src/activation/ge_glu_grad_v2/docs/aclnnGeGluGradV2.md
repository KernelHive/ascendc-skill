# aclnnGeGluGradV2

## 支持的产品型号
- Atlas A2 训练系列产品。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnGeGluGradV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGeGluGradV2”接口执行计算。

- `aclnnStatus aclnnGeGluGradV2GetWorkspaceSize(const aclTensor *dy, const aclTensor *x, const aclTensor *gelu, int64_t dim, int64_t approximate, bool activateLeft, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnGeGluGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述
- 算子功能：完成GeGluGradV2的反向。

## aclnnGeGluV2GetWorkspaceSize
- **参数说明**：
  
  - dy（aclTensor*，计算输入）：npu device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT32，shape中除dim维外，其它维的大小跟x一样，dim维的大小是x的一半，支持非连续的Tensor，支持空Tensor。数据格式支持ND。
  - x（aclTensor*，计算输入）：表示待计算的数据，公式中的x，Device侧的aclTensor，数据类型支持FLOAT32、FLOAT16、BFLOAT16，支持非连续的Tensor，支持空Tensor。数据格式支持ND。
  - Atlas 800训练系列产品：数据类型支持BFLOAT16、FLOAT16。
  - gelu（aclTensor*，计算输入）：npu device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT32，shape需要与dy一样，支持非连续的Tensor，支持空Tensor。数据格式支持ND6、FLOAT32。
  - dim：可选入参，设定的slice轴，数据类型支持INT64。
  - approximate：可选入参，表示Gelu的算法选择 erf/tanh，数据类型支持INT64。
  - activateleft：可选入参，表示激活函数操作数据块方向，默认false, 对右边做activate，数据类型支持BOOL。
  - out（aclTensor*，计算输出）：计算输出，npu device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT32，shape需要与x一样，支持非连续的Tensor，支持空Tensor。数据格式支持ND。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。  

- **返回值**：
  aclnnStatus：返回状态码。
  
  ```
  161001（ACLNN_ERR_PARAM_NULLPTR）：1. 参数dy、x、gelu、out是空指针。
  161002（ACLNN_ERR_PARAM_INVALID）：1. 参数dy、x、gelu、out的数据类型不在支持的范围内。
                                    2. 参数dy、x、gelu、out的shape不满足要求。
                                    3. 当x.dim()==0时，参数dim的取值范围不在[-1, 0]内；当x.dim()>0时，参数dim的取值范围不在[-x.dim(), x.dim()-1]内。
                                    4. 参数approximate的取值范围不是0和1。
  ```

## aclnnGeGluGradV2
- **参数说明**：
  - workspace(void*，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnGeGluV2GetWorkspaceSize获取。
  - executor(aclOpExecutor*，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的AscendCL Stream流。

- **返回值**：
  aclnnStatus：返回状态码。

## 约束与限制
无。