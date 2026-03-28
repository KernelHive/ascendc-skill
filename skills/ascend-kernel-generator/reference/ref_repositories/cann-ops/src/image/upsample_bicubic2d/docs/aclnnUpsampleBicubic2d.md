# aclnnUpsampleBicubic2d

## 支持的产品型号

- Atlas 200I/500 A2 推理产品。
- Atlas 推理系列产品。
- Atlas 训练系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnUpsampleBicubic2dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleBicubic2d”接口执行计算。

- `aclnnStatus aclnnUpsampleBicubic2dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize, const bool alignCorners, const double scalesH, const double scalesW, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnUpsampleBicubic2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

- 算子功能：对由多个输入通道组成的输入信号应用2D双三次上采样。如果输入tensor x的shape为(N, C, H, W) ，则输出tensor out的shape为(N, C, outputSize[0], outputSize[1])。
- 计算公式：对于一个二维插值点$(N, C, h, w)$，插值$out(N, C, h, w)$可以表示为：

  $$
  {out(N, C, h, w)}=\sum_{i=0}^{3}\sum_{j=0}^{3}{W(i, j)}*{f(h_i, w_j)}
  $$

  $$
  scaleH =\begin{cases}
  (self.dim(2)-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  self.dim(2) / outputSize[0] & otherwise
  \end{cases}
  $$

  $$
  scaleW =\begin{cases}
  (self.dim(3)-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  self.dim(3) / outputSize[1] & otherwise
  \end{cases}
  $$

  其中：
  - $f(h_i, w_j)$是原图像在$(h_i, w_j)$的像素值。
  - $W(i, j)$是双三次抗锯齿插值的权重，定义为：

    $$
    W(d) =\begin{cases}
    (a+2)|d|^3-(a+3)|d|^2+1 & |d|\leq1 \\
    a|d|^3-5a|d|^2+8a|d|-4a & 1<|d|<2 \\
    0 & otherwise
    \end{cases}
    $$

    其中：
    - $a=-0.75$
    - $d = |(h, w) - (h_i, w_j)|$

## aclnnUpsampleBicubic2dGetWorkspaceSize

- **参数说明**：

  - self（aclTensor*，计算输入）：Device侧的aclTensor。表示进行上采样的输入张量。支持非连续的Tensor，不支持空Tensor。数据格式支持NCHW、ND（当数据格式为ND时，默认按照NCHW格式处理）。shape支持4维，数据类型需要与出参`out`的数据类型一致。
    - Atlas 200I/500 A2 推理产品、Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - outputSize（aclIntArray*，计算输入）：指定输出空间大小，Host侧的aclIntArray，数据类型支持INT64。size需要等于2，且各元素均大于0。
  - alignCorners（bool，计算输入）：决定是否对齐角像素点，Host侧的bool。alignCorners为True，则输入和输出张量的角像素点会被对齐，否则不对齐。
  - scalesH（double，计算输入）：指定空间大小的height维度乘数，Host侧的double型。
  - scalesW（double，计算输入）：指定空间大小的width维度乘数，Host侧的double型。
  - out（aclTensor*，计算输出）：Device侧的aclTensor。表示采样后的输出张量。支持非连续的Tensor，不支持空Tensor。数据格式支持NCHW、ND。shape支持4维：(batch, channel, height, width)，其中batch与channel分别来源于入参`self`的第零维和第一维，height与width分别来源于`outputSize`的第一与第二个值。数据类型和数据格式需要与`self`的数据类型和数据格式一致。
    - Atlas 200I/500 A2 推理产品、Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - workspaceSize（uint64_t*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码。

```
第一段接口完成入参校验，出现以下场景时报错：
返回161001 （ACLNN_ERR_PARAM_NULLPTR）:1. 传入的self、outputSize或out是空指针。
返回161002 （ACLNN_ERR_PARAM_INVALID）:1. self或out的数据类型不在支持的范围之内。
                                      2. self和out的数据类型不一致。
                                      3. self的shape不是4维。
                                      4. self在C、H、W维度上的size不大于0。
                                      5. outputSize的size大小不等于2。
                                      6. outputSize的某个元素值不大于0。
```

## aclnnUpsampleBicubic2d

- **参数说明**：

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleBicubic2dGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  **aclnnStatus**：返回状态码。

## 约束与限制

参数outputSize与参数scalesH、scalesW，在使用时二选一，即：
- 当入参scalesH和入参scalesW的值都为0时，使用入参outputSize的参数值。
- 当入参scalesH和入参scalesW的值不都为0时，使用入参scalesH和入参scalesW的参数值，且$outputSize=[floor(selfH*scalesH)，floor(selfW*scalesW)]$。
