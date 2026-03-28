# aclnnGeGluV2

## 支持的产品型号
- Atlas 推理系列产品。
- Atlas 训练系列产品。
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnGeGluV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGeGluV2”接口执行计算。

- `aclnnStatus aclnnGeGluV2GetWorkspaceSize(const aclTensor *self, int64_t dim, int64_t approximate, bool activate_left, aclTensor *out, aclTensor *outGelu, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnGeGluV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述
- 算子功能：GeGluV2实现了一个门控线性单元（Gated Linear Unit, GLU）的变体，它使用GELU（Gaussian Error Linear Unit）作为激活函数，实现x的GeGluV2计算。 
- 计算公式：  
  <p style="text-align: center">
  GeGluV2<sub>x,W,V,b,c</sub> = Gelu<sub>xW+b</sub> * (xV + c)
  </p>

## aclnnGeGluV2GetWorkspaceSize
- **参数说明**：
  
  - self（aclTensor*，计算输入）：表示待计算的数据，公式中的x，Device侧的aclTensor，数据类型支持FLOAT32、FLOAT16、BFLOAT16，支持非连续的Tensor，支持空Tensor。数据格式支持ND。shape维度不高于8维。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
  - dim：可选入参，设定的slice轴，数据类型支持INT64。
  - approximate：可选入参，表示Gelu的算法选择 erf/tanh，数据类型支持INT64。
  - activate_left：可选入参，表示激活函数操作数据块方向，默认false，对右边做activate，数据类型支持BOOL。
  - out（aclTensor*，计算输出）：表示计算结果，Device侧的aclTensor，数据类型和shape与计算输入self的一致，支持非连续的Tensor，支持空Tensor。数据格式支持ND。shape维度不高于8维。
  - outGelu（aclTensor*，计算输出）：表示计算结果，Device侧的aclTensor，数据类型和shape与计算输入self的一致，支持非连续的Tensor，支持空Tensor。数据格式支持ND。shape维度不高于8维。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。  

- **返回值**：
  aclnnStatus：返回状态码。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：参数self、out、outGelu是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. 参数self、out、outGelu的数据类型不在支持的范围内。
                                        2. 参数out、outGelu的数据类型与self不一致。
                                        3. self、out、outGelu的维数大于8。
                                        4. 当self.dim()==0时，dim的取值不在[-1, 0]范围内；当dimself.dim()>0时，取值不在[-self.dim(), self.dim()-1]范围内。
                                        5. out、outGelu在dim维的size不等于self在dim维size的1/2。
  ```

## aclnnGeGluV2
- **参数说明**：
  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnGeGluV2GetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：
  aclnnStatus：返回状态码。

## 约束与限制
无。