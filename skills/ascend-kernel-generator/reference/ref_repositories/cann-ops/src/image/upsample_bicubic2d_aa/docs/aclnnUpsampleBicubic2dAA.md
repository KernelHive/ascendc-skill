# aclnnUpsampleBicubic2dAA

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnUpsampleBicubic2dAAGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleBicubic2dAA”接口执行计算。
- `aclnnStatus aclnnUpsampleBicubic2dAAGetWorkspaceSize(const aclTensor* x, const aclIntArray* outputSize, const bool alignCorners, const double scalesH, const double scalesW, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnUpsampleBicubic2dAA(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：对由多个输入通道组成的输入信号应用双三次抗锯齿算法进行上采样。如果输入tensor x的shape为(N, C, H, W) ，则输出tensor out的shape为(N, C, outputSize[0], outputSize[1])。
- 计算公式：对于一个二维插值点$(N, C, h, w)$，插值$out(N, C, h, w)$可以表示为：

  $$
  {out(N, C, h, w)}=\sum_{i=0}^{kW}\sum_{j=0}^{kH}{W(i, j)}*{f(h_i, w_j)}
  $$

  $$
  scaleH =\begin{cases}
  (x.dim(2)-1 / outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  x.dim(2) / outputSize[0] & otherwise
  \end{cases}
  $$
  
  $$
  scaleW =\begin{cases}
  (x.dim(3)-1 / outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  x.dim(3) / outputSize[1] & otherwise
  \end{cases}
  $$

  其中：
  - 如果$scaleH >= 1$， 则$kH = 1/scaleH$，否则$kH = 4$
  - 如果$scaleW >= 1$， 则$kW = 1/scaleW$，否则$kW = 4$
  - $h_i = |h| + i$
  - $w_j = |w| + j$
  - $f(h_i, w_j)$是原图像在$(h_i, w_j)$的像素值
  - $W(i, j)$是双三次抗锯齿插值的权重，定义为：

    $$
    W(d) =\begin{cases}
    (a+2)|d|^3-(a+3)|d|^2+1 & |d|\leq1 \\
    a|d|^3-5a|d|^2+8a|d|-4a & 1<|d|<1 \\
    0 & otherwise
    \end{cases}
    $$

    其中：
    - 抗锯齿场景$a=-0.5$。
    - $d = |(h, w) - (h_i, w_j)|$

## aclnnUpsampleBicubic2dAAGetWorkspaceSize

- **参数说明**：

  - x（aclTensor\*，计算输入）：Device侧的aclTensor。表示进行上采样的输入张量。数据类型支持FLOAT、FLOAT16、BFLOAT16。支持非连续的Tensor，不支持空Tensor。数据格式支持NCHW、ND（当数据格式为ND时，默认按照NCHW格式处理），shape维度仅支持4维shape的tensor，数据类型与出参`out`的数据类型一致。
  - outputSize（aclIntArray\*，计算输入）：Device侧的aclIntArray，指定输出空间大小，数据类型支持INT64。
  - alignCorners（bool，计算输入）：Host侧的BOOL型参数，指定是否对齐角像素点。如果为True，则输入和输出张量的角像素点会被对齐，否则不对齐，默认为False。
  - scalesH（double，计算输入）：Host侧的DOUBLE型参数，指定空间大小的height维度乘数。
  - scalesW（double, 计算输入）：Host侧的DOUBLE型参数，指定空间大小的width维度乘数。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor。表示采样后的输出张量。数据类型支持FLOAT、FLOAT16、BFLOAT16。支持非连续的Tensor，不支持空Tensor。数据格式支持NCHW、ND，shape维度仅支持4维shape的tensor，数据类型与入参`x`的数据类型一致。
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码。

```
第一段接口完成入参校验，出现以下场景时报错：
返回161001 (ACLNN_ERR_PARAM_NULLPTR)： 1. 传入的x、outputSize或out是空指针。
返回161002 (ACLNN_ERR_PARAM_INVALID)： 1. x或out的数据类型不在支持的范围之内。
                                      2. x和out的数据类型不一致。
                                      3. x的shape不是4维。
```

## aclnnUpsampleBicubic2dAA

- **参数说明**：

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleBicubic2dAAGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  aclnnStatus：返回状态码。

## 约束与限制

- 输入数据缩放场景缩小倍数必须小于等于50，即输入shape的高度H/outputSize[0]以及宽度W/outputSize[1]必须小于等于50。
- 参数outputSize与参数scalesH、scalesW，在使用时二选一，即：
  - 当入参scalesH和入参scalesW的值都为0时，使用入参outputSize的参数值。
  - 当入参scalesH和入参scalesW的值不都为0时，使用入参scalesH和入参scalesW的参数值，且$outputSize=[floor(xH*scalesH)，floor(xW*scalesW)]$。

