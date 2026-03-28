# aclnnUpsampleBilinear2d

## 支持的产品型号
- Atlas 推理系列产品
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnUpsampleBilinear2dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleBilinear2d”接口执行计算。

- `aclnnUpsampleBilinear2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize, const bool alignCorners, const double scalesH, const double scalesW, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnUpsampleBilinear2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：对由多个输入通道组成的输入信号应用2D双线性上采样。
  - 对于输入shape：
  如果输入shape为(N，C，H，W)，则输出shape为(N，C，outputSize[0]，outputSize[1])；
  - 对于中心对齐的选择：一般像素被视为网格。当alignCorners = True时，像素被视为网格左上角的点，输出拐角处的像素与原图像的拐角像素中心对齐，同方向点之间是等间距的；当alignCorners = False时, 像素被视为网格的交叉线上的点，输出拐角处的像素依然是原图像的拐角像素，但同方向点之间是不等距的。
- 示例：(1，1，3，3) -> (1，1，5，5)

$$
\begin{matrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{matrix} \tag{原图}
$$

$$
\begin{matrix}
   1 & 1.4 & 2 & 2.6 & 3 \\
   2.2 & 2.6 & 3.2 & 3.8 & 4.2 \\
   4 & 4.4 & 5 & 5.6 & 6 \\
   5.8 & 6.2 & 6.8 & 7.4 & 7.8 \\
   7 & 7.4 & 8 & 8.6 & 9 
  \end{matrix}  \tag{alignCorners = False}
$$

$$
\begin{matrix}
   1 & 1.5 & 2 & 2.5 & 3 \\
   2.5 & 3 & 3.5 & 4 & 4.5 \\
   4 & 4.5 & 5 & 5.5 & 6 \\
   5.5 & 6 & 6.5 & 7 & 7.5 \\
   7 & 7.5 & 8 & 8.5 & 9 
  \end{matrix}  \tag{alignCorners = True}
$$

## aclnnUpsampleBilinear2dGetWorkspaceSize

- **参数说明**：
  - self（aclTensor\*，计算输入）：Device侧的aclTensor，表示进行上采样的输入张量。支持非连续的Tensor，不支持空Tensor。shape仅支持4维，数据格式支持NCHW和NHWC。当数据类型为DOUBLE时，仅支持NHWC格式。
    - Atlas 推理系列产品：数据类型支持FLOAT、FLOAT16、DOUBLE。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT、BFLOAT16、FLOAT16、DOUBLE。 
  - outputSize（aclIntArray\*，计算输入）：Host侧的aclIntArray，输出空间大小。
  - alignCorners（bool，计算输入）：Host侧的bool类型参数。如果设置为True，则输入和输出张量按其角像素的中心点对齐，保留角像素处的值；如果设置为False，则输入和输出张量通过其角像素的角点对齐，并使用边缘值对边界外的值进行填充。
  - scalesH（double，计算输入）：Host侧的double常量，空间大小的height维度乘数，不能传入负值。
  - scalesW（double，计算输入）：Host侧的double常量，空间大小的width维度乘数，不能传入负值。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，表示采样后的输出张量。shape仅支持4维，支持非连续的Tensor，不支持空Tensor。数据格式支持NCHW和NHWC。当数据类型为DOUBLE时，仅支持NHWC格式。数据类型和数据格式与入参`self`的数据类型和数据格式保持一致。
    - Atlas 推理系列产品：数据类型支持FLOAT、FLOAT16、DOUBLE。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT、BFLOAT16、FLOAT16、DOUBLE。 
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值**：

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回561103 （ACLNN_ERR_INNER_NULLPTR）：1. API内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。
  返回161002 （ACLNN_ERR_PARAM_INVALID）：1. self的数据类型不在支持的范围之内。
                                        2. scalesH/scalesW的值为负值。
                                        3. self和out的N/C轴的维度大小不相等。
                                        4. self和out的数据格式不在支持的范围之内。
  ```

## aclnnUpsampleBilinear2d

- **参数说明**：
  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleBilinear2dGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。
- **返回值**：

aclnnStatus：返回状态码。

## 约束与限制

参数outputSize与参数scalesH、scalesW，在使用时二选一，即：
- 当入参scalesH和入参scalesW的值都为0时，使用入参outputSize的参数值。
- 当入参scalesH和入参scalesW的值不都为0时，使用入参scalesH和入参scalesW的参数值，且$outputSize=[floor(selfH*scalesH)，floor(selfW*scalesW)]$。

