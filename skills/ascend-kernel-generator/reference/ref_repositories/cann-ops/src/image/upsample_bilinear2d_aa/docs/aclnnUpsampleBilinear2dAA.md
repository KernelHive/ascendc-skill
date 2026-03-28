# aclnnUpsampleBilinear2dAA

## 支持的产品型号
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnUpsampleBilinear2dAAGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleBilinear2dAA”接口执行计算。

- `aclnnStatus aclnnUpsampleBilinear2dAAGetWorkspaceSize(const aclTensor *input, const aclIntArray *outputSize, bool alignCorners, double scalesH, double scalesW, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnUpsampleBilinear2dAA(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：对由多个输入通道组成的输入信号应用2D双线性抗锯齿采样。
- 计算公式：对于一个二维插值点$(N, C, H, W)$, 插值$I(N, C, H, W)$可以表示为：

  $$
  {I(N, C, H, W)} = \sum_{i=0}^{kW}\sum_{j=0}^{kH}{w(i) * w(j)} * {f(h_i, w_j)}/\sum_{i=0}^{kW}w(i)/\sum_{j=0}^{kH}w(j)
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
  - $kW$、$kH$分别表示W方向和H方向影响插入点大小的点的数量
  - 如果$scaleH >= 1$， 则$kH = floor(scaleH) * 2 + 1$，否则$kH = 3$
  - 如果$scaleW >= 1$， 则$kW = floor(scaleW) * 2 + 1$，否则$kW = 3$
  - $f(h_i, w_j)$是原图像在$(h_i, w_j)$的像素值
  - $w(i)$、$w(j)$是双线性抗锯齿插值的W方向和H方向权重，计算公式为：

    $$
      w(i) = \begin{cases}
      1 - |h_i - h| & |h_i -h| < 1 \\
      0 & otherwise
      \end{cases}
    $$

    $$
      w(j) = \begin{cases}
      1 - |w_j - w| & |w_i -w| < 1 \\
      0 & otherwise
      \end{cases}
    $$
  

## aclnnUpsampleBilinear2dAAGetWorkspaceSize

* **参数说明**：
  - input（aclTensor\*，计算输入）：Device侧的aclTensor，表示进行采样的输入张量。数据类型支持FLOAT、FLOAT16、BFLOAT16。支持非连续的Tensor，不支持空Tensor。数据格式支持NCHW、ND（当数据格式为ND时，默认按照NCHW格式处理）。shape维度仅支持4维的Tensor。
  - outputSize（aclIntArray\*，计算输入）：Host侧的aclIntArray，输出空间大小，要求是二维数组，数据类型支持INT64，取值与出参`out`的H、W维度一样。
  - alignCorners（bool，计算输入）：Host侧bool类型参数，指定是否是角对齐。如果设置为`true`，则输入和输出张量按其角像素的中心点对齐，保留角像素处的值。如果设置为`false`，则输入和输出张量通过其角像素的角点对齐，并使用边缘值对边界外的值进行填充。
  - scalesH（double，计算输入）: Host侧double类型参数，空间大小的height维度乘数。
  - scalesW（double，计算输入）: Host侧double类型参数，空间大小的width维度乘数。
  - out（aclTensor\*，计算输出）: Device侧的aclTensor，表示采样后的输出张量。数据类型支持FLOAT、FLOAT16、BFLOAT16。支持非连续的Tensor，不支持空Tensor。数据格式支持NCHW、ND。shape维度仅支持4维的Tensor。数据类型和数据格式与入参`input`的数据类型和数据格式保持一致。
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
* **返回值**：

  aclnnStatus：返回状态码。
```
第一段接口完成入参校验，出现以下场景时报错：
返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的input、outputSize或out是空指针。
返回161002（ACLNN_ERR_PARAM_INVALID）: 1. input或out的数据类型不在支持的范围之内。
                                      2. input和out的数据类型不一致。
                                      3. input的shape不是4维。
```


## aclnnUpsampleBilinear2dAA

* **参数说明**：
  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleBilinear2dAAGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。
* **返回值**：

aclnnStatus：返回状态码。

## 约束与限制

- 输入数据缩放场景缩小倍数必须小于等于50，即输入shape的高度H/outputSize[0]以及宽度W/outputSize[1]必须小于等于50。
- 参数outputSize与参数scalesH、scalesW，在使用时二选一，即：
  - 当入参scalesH和入参scalesW的值都为0时，使用入参outputSize的参数值。
  - 当入参scalesH和入参scalesW的值不都为0时，使用入参scalesH和入参scalesW的参数值，且$outputSize=[floor(inputH*scalesH)，floor(inputW*scalesW)]$。

