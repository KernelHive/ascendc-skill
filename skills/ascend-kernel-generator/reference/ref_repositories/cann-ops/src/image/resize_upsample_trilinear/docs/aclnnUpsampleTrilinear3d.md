# aclnnUpsampleTrilinear3d

## 支持的产品型号

- Atlas 推理系列产品。
- Atlas 训练系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnUpsampleTrilinear3dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleTrilinear3d”接口执行计算。

- `aclnnStatus aclnnUpsampleTrilinear3dGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize, bool alignCorners, double scalesD, double scalesH, double scalesW, aclTensor *out,  uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnUpsampleTrilinear3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：对由多个输入通道组成的输入信号应用三线性插值算法进行上采样。
- 计算公式：
  - 核心算法逻辑：
    1.将目标图像缩放到和原始图像一样大的尺寸。
    2.计算缩放之后的目标图像的点，以及前后相邻的原始图像的点。
    3.分别计算相邻点到对应目标点的权重，按照权重相乘累加即可得到目标点值。
  - 具体计算逻辑：
    缩放方式分为角对齐和边对齐，角对齐表示按照原始图片左上角像素中心点对齐，边对齐表示按照原始图片左上角顶点及两条边对齐，在计算缩放系数和坐标位置时有不同。记inputsize和outputsize分别为input和output对应维度方向上的shape大小。则有以下公式：

    $$
    scale =\begin{cases}
    (inputsize-1) / (outputsize-1) & alignCorners=true \\
    inputsize / outputsize & alignCorners=false
    \end{cases}
    $$

    那么，对于output的某个方向上面的点p(x,y)，映射回原始图像中的点记为q(x',y')，则有关系: 
    
    $$
    x' =\begin{cases}
    x * scale & alignCorners=true \\
    MAX(0,{(x+0.5)*scale-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    y' =\begin{cases}
    y * scale & alignCorners=true \\
    MAX(0,{(y+0.5)*scale-0.5}) & alignCorners=false
    \end{cases}
    $$

    以x方向为例，记：

    $$
    x_{0} =int(x'),x_{1} =int(x')+1, lamda_{0} = x'-x_{0}, lamda_{1} = 1-lamda_{0}
    $$
    
    则有以下公式：

    $$
    {V(p_{x})} = {V(p_{x0})} * {lamda_{0}} + {V(p_{x1})} * {lamda_{1}}
    $$

## aclnnUpsampleTrilinear3dGetWorkspaceSize

- **参数说明**：

  - self（aclTensor*，计算输入）：Device侧的aclTensor。表示进行上采样的输入张量。支持非连续的Tensor，不支持空Tensor。数据格式支持NCDHW、NDHWC、ND（当数据格式为ND时，默认按照NCDHW格式处理）。shape仅支持五维Tensor。
    - Atlas 推理系列产品：数据类型支持FLOAT、FLOAT16，不支持inf、-inf输入。
    - Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16、DOUBLE。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件：数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE。
  - outputSize（aclIntArray*,计算输入）：Host侧的aclIntArray，数据类型支持INT64，指定输出Tensor大小，size大小为3，表示出参`out`在D、H和W维度上的空间大小。
  - alignCoreners（bool，计算输入）：Host侧的bool类型参数，指定是否对齐角像素点。如果为true，则输入和输出张量的角像素点会被对齐，否则不对齐。
  - scalesD（double，计算输入）：Host侧的double常量，表示输出`out`的depth维度乘数。
  - scalesH（double，计算输入）：Host侧的double常量，表示输出`out`的height维度乘数。
  - scalesW（double，计算输入）：Host侧的double常量，表示输出`out`的width维度乘数。
  - out（aclTensor*，计算输出）：Device侧的aclTensor，表示采样后的输出张量。支持非连续的Tensor，不支持空Tensor。数据格式支持NCDHW、NDHWC、ND。shape仅支持五维Tensor。数据类型和数据格式与入参`self`的数据类型和数据格式保持一致。
    - Atlas 推理系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16、DOUBLE。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件：数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码。

```
第一段接口完成入参校验，出现以下场景时报错：
返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的 self 、outputSize或out是空指针时。
返回161002（ACLNN_ERR_PARAM_INVALID）: 1. self的数据类型不在支持的范围之内或self与out数据类型不同。
                                      2. self 的shape不是5维。
                                      3. outputSize的size大小不等于3。
                                      4. self在D、H、W维度上的size不大于0。
                                      5. outputSize的某个元素值不大于0。
                                      6. self的C维度为0。
```

## aclnnUpsampleTrilinear3d

- **参数说明**：

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleTrilinear3dGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  **aclnnStatus**：返回状态码。

## 约束与限制

无。
