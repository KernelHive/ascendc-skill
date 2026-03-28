# aclnnUpsampleNearest3d

## 支持的产品型号

- Atlas 推理系列产品。
- Atlas 训练系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnUpsampleNearest3dGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnUpsampleNearest3d”接口执行计算。

- `aclnnStatus aclnnUpsampleNearest3dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize, double scalesD, double scalesH, double scalesW, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnUpsampleNearest3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

- 算子功能：对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。
- 计算公式：
  - 核心算法逻辑：
    1.将目标图像缩放到和原始图像一样大的尺寸。
    2.对于缩放之后的目标图像的点，计算距离最近的原始图像的点，后者的值直接复制给前者。
  - 具体计算逻辑：
    记inputsize和outputsize分别为self和out对应维度方向上的shape大小。则计算缩放系数有以下公式：

    $$
    scale = inputsize / outputsize
    $$

    那么，对于out的某个方向上面的点p(x,y,z)，映射回原始图像中的点记为q(x',y',z')，则有关系: 
    
    $$
    x' = \min(\lfloor x * scale \rfloor, inputsize - 1)
    $$
    $$
    y' = \min(\lfloor y * scale \rfloor, inputsize - 1)
    $$
    $$
    z' = \min(\lfloor z * scale \rfloor, inputsize - 1)
    $$
  
    则有以下公式：

    $$
    {V(p_{x,y,z})} = {V(q_{x',y',z'})}
    $$


## aclnnUpsampleNearest3dGetWorkspaceSize

- **参数说明**

  - self（aclTensor*，计算输入）：Device侧的aclTensor，表示进行上采样的输入张量。shape仅支持五维，支持非连续的Tensor，不支持空Tensor。数据格式支持NCDHW、NDHWC、ND（当数据格式为ND时，默认按照NCDHW格式处理）。
    - Atlas 推理系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16、DOUBLE。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE。
  - outputSize（aclIntArray*，计算输入）：指定输出`out`的Tensor大小，Host侧的aclIntArray。size大小为3，各元素均大于零。表示输入self在D、H和W维度上的空间大小。
  - scalesD（double，计算输入）：指定输出`out`的depth维度乘数，Host侧的DOUBLE型。
  - scalesH（double，计算输入）：指定输出`out`的height维度乘数，Host侧的DOUBLE型。
  - scalesW（double，计算输入）：指定输出`out`的width维度乘数，Host侧的DOUBLE型。
  - out（aclTensor、*，计算输出）：Device侧的aclTensor，表示采样后的输出张量。shape仅支持五维，支持非连续的Tensor，不支持空Tensor。数据格式支持NCDHW、NDHWC、ND。数据类型和数据格式需与入参`self`的数据类型和数据格式保持一致。
    - Atlas 推理系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16、DOUBLE。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。
  
- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的 self 、outputSize或out是空指针时。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self的数据类型不在支持的范围之内或self与out数据类型不同。
                                        2. self的shape不是5维。
                                        3. outputSize的size大小不等于3。
                                        4. self在D、H、W维度上的size不大于0。
                                        5. outputSize的某个元素值不大于0。
                                        6. self的C维度为0。
                                        7. out的shape不等于由self和outputSize推导得到shape。
  ```

## aclnnUpsampleNearest3d

- **参数说明**

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleNearest3dGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

参数outputSize与参数scalesD、scalesH、scalesW，在使用时二选一，即：
- 当入参scalesD、scalesH、scalesW的值都为0时，使用入参outputSize的参数值。
- 当入参scalesD、scalesH、scalesW的值不都为0时，使用入参scalesD、scalesH、scalesW的参数值，且$outputSize=[floor(selfD*scalesD)，floor(selfH*scalesH)，floor(selfW*scalesW)]$。
