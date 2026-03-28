# aclnnConvolutionBackward

## 支持的产品型号
- 昇腾910B AI处理器。
- 昇腾910_93 AI处理器。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnConvolutionBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnConvolutionBackward”接口执行计算。

- `aclnnStatus aclnnConvolutionBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *input, const aclTensor *weight, const aclIntArray *biasSizes, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool transposed, const aclIntArray *outputPadding, int groups, const aclBoolArray *outputMask, int8_t cubeMathType, aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *workspaceSize, aclOpExecutor **executor)`

- `aclnnStatus aclnnConvolutionBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## 功能描述

- 算子功能：卷积的反向传播。根据输出掩码设置计算输入、权重和偏差的梯度。此函数支持1D、2D或3D卷积。
- 计算公式：

  卷积反向传播需要计算对卷积正向的输入张量 $x$、卷积核权重张量 $w$ 和偏置 $b$ 的梯度。

  对于 $x$ 的梯度 $\frac{\partial L}{\partial x}$：

  $$
  \frac{\partial L}{\partial x_{n, c_{in}, i, j}} = \sum_{c_{out}=1}^{C_{out}} \sum_{p=1}^{k_H} \sum_{q=1}^{k_W} \frac{\partial L}{\partial y_{n, c_{out}, i-p, j-q}}\cdot w_{c_{out}, c_{in}, p, q}
  $$

  其中，$L$ 为损失函数，$\frac{\partial L}{\partial y}$ 为输出张量 $y$ 对 $L$ 的梯度。

  对于 $w$ 的梯度 $\frac{\partial L}{\partial w}$：

  $$
  \frac{\partial L}{\partial w_{c_{out}, c_{in}, p, q}} = \sum_{n=1}^{N} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} x_{n, c_{in}, i \cdot s_H + p, j \cdot s_W + q} \cdot \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
  $$

  对于 $b$ 的梯度 $\frac{\partial L}{\partial b}$：

  $$
  \frac{\partial L}{\partial b_{c_{out}}} = \sum_{n=1}^{N}       \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
  $$


## aclnnConvolutionBackwardGetWorkspaceSize

- **参数说明：**

  * gradOutput(aclTensor *，计算输入)：公式中的$\frac{\partial L}{\partial y}$，shape不支持broadcast，要求和input、weight满足卷积输入输出shape的推导关系。其数据类型与input、weight满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)和[约束与限制](#约束与限制)）。支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持NCL、NCHW、NCDHW，且需要与input、weight一致。不支持空tensor。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT、FLOAT16。1d、2d和3d transposed=false场景，各个维度的大小应该大于等于1。
  * input(aclTensor *，计算输入)：公式中的$x$，shape不支持broadcast，要求和gradOutput、weight满足卷积输入输出shape的推导关系。其数据类型与gradOutput、weight满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)和[约束与限制](#约束与限制)）。支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持NCL、NCHW、NCDHW，且需要与gradOutput、weight一致。仅支持N或C维度为0的空tensor。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT、FLOAT16。1d、2d和3d transposed=false场景，各个维度的大小应该大于等于1。
  * weight(aclTensor *，计算输入)：公式中的$w$，shape不支持broadcast，要求和gradOutput、input满足卷积输入输出shape的推导关系。其数据类型与gradOutput、input满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)和[约束与限制](#约束与限制)）。支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持NCL、NCHW、NCDHW，且需要与gradOutput、input一致。不支持空tensor。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT、FLOAT16。2d和3d transposed=false场景，H、W的大小应该在[1,255]的范围内，其他维度的大小应该大于等于1。1d transposed=false场景，L的大小应该在[1,255]的范围内，其他维度的大小应该大于等于1。
  * biasSizes(aclIntArray *，计算输入)：卷积正向过程中偏差(bias)的shape。数据类型为int64，数组长度是1。 其在普通卷积中等于[weight.shape[0]],在转置卷积中等于[weight.shape[1] * groups]。空Tensor场景下，当outputMask指定偏差的梯度需要计算时，biasSizes不能为nullptr。
  * stride(aclIntArray *，计算输入)：反向传播过程中卷积核在输入上移动的步长。数据类型为int64，数组长度为weight维度减2，数值必须大于0。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：3d transposed=false场景，strideD应该大于等于1，strideH、strideW应该在[1,63]的范围内。1d和2d transposed=false场景，各个值都应该大于等于1。
  * padding(aclIntArray *，计算输入)：反向传播过程中对于输入填充。数据类型为int64，数组长度可以为weight维度减2，在2d场景下数组长度可以为4。数值必须大于等于0。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：3d transposed=false场景，paddingD应该大于等于0，paddingH、paddingW应该在[0,255]的范围内。1d和2d transposed=false场景，各个值都应该在[0,255]的范围内。
  * dilation(aclIntArray *，计算输入)：反向传播过程中的膨胀参数。数据类型为int64，数组长度可以为weight维度减2。数值必须大于0。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：1d、2d和3d transposed=false场景，各个值都应该在[1,255]的范围内。
  * transposed(bool，计算输入)：转置卷积使能标志位, 当其值为True时使能转置卷积。
  * outputPadding(aclIntArray *，计算输入)：反向传播过程中对于输出填充, 仅在transposed为True时生效。数据类型为int64，数组长度可以为weight维度减2，数值必须大于等于0且小于stride。transposed为False时，仅支持outputPadding为0。
  * groups(int，计算输入)：反向传播过程中输入通道的分组数。 数据类型为int, 数值必须大于0, groups*weight的C维度=input的C维度。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：1d、2d和3d transposed=false场景，groups应该在[1,65535]的范围内。
  * outputMask(const aclBoolArray *，计算输入)：输出掩码参数, 指定输出中是否包含输入、权重、偏差的梯度。反向传播过程输出掩码参数为True对应位置的梯度。
  * cubeMathType(int8_t，计算输入)：用于判断Cube单元应该使用哪种计算逻辑进行运算，INT8类型的枚举值，枚举如下：
    * 0:KEEP_DTYPE，保持输入的数据类型进行计算。
      - 昇腾310P AI处理器、昇腾910 AI处理器：当输入是FLOAT，Cube计算单元暂不支持，取0时会报错。
    * 1:ALLOW_FP32_DOWN_PRECISION，允许将输入数据降精度计算。
      - 昇腾310P AI处理器、昇腾910 AI处理器：当输入是FLOAT，转换为FLOAT16计算。当输入为其他数据类型时不做处理。
      - 昇腾910B AI处理器、昇腾910_93 AI处理器：当输入是FLOAT，转换为HFLOAT32计算。当输入为其他数据类型时不做处理。
    * 2:USE_FP16，允许转换为数据类型FLOAT16进行计算。当输入数据类型是FLOAT，转换为FLOAT16计算。
      - 昇腾910B AI处理器、昇腾910_93 AI处理器：当输入是BFLOAT16时不支持该选项。
    * 3:USE_HF32，允许转换为数据类型HFLOAT32计算。当输入是FLOAT16，仍使用FLOAT16计算。
      - 昇腾310P AI处理器、昇腾910 AI处理器：当输入是FLOAT，Cube计算单元暂不支持。
      - 昇腾910B AI处理器、昇腾910_93 AI处理器：当输入是FLOAT，转换为HFLOAT32计算。当输入为其他数据类型时不支持该选项。
  * gradInput(aclTensor *, 计算输出)：公式中的$\frac{\partial L}{\partial x}$，[数据格式](common/数据格式.md)为NCL，NCHW、NCDHW，且与input一致。数据类型与input保持一致。
    - 昇腾310P AI处理器、昇腾910 AI处理器：数据类型支持FLOAT、FLOAT16。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT、FLOAT16。
  * gradWeight(aclTensor *, 计算输出)：公式中的$\frac{\partial L}{\partial w}$，[数据格式](common/数据格式.md)为NCL，NCHW、NCDHW，且与input一致。数据类型与weight保持一致。
    - 昇腾310P AI处理器、昇腾910 AI处理器：数据类型支持FLOAT、FLOAT16。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT、FLOAT16。
  * gradBias(aclTensor *, 计算输出)：公式中的$\frac{\partial L}{\partial b}$，且数据类型与gradOutput一致，[数据格式](common/数据格式.md)为ND。
    - 昇腾310P AI处理器、昇腾910 AI处理器：数据类型支持FLOAT、FLOAT16。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT、FLOAT16。
  * workspaceSize(uint64_t *, 出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor **, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 ACLNN_ERR_PARAM_NULLPTR: 1. 传入的gradOutput、input、weight、biasSizes、stride、padding、dilation、outputPadding、outputMask、gradInput、gradWeight是空指针。
                                  2. 输出中包含偏差的梯度时，传入的gradBias是空指针。
  161002 ACLNN_ERR_PARAM_INVALID: 1. gradOutput、input、weight的数据类型不在支持的范围之内。
                                  2. gradOutput、input、weight的数据格式不在支持的范围之内。
                                  3. gradOutput、input、weight的shape不符合约束。
                                  4. biasSizes、stride、padding、dilation、outputPadding的shape不符合约束。
                                  5. 不符合groups*weight的C维度=input的C维度。
                                  6. 当前处理器不支持卷积反向传播。
  561103 ACLNN_ERR_INNER_NULLPTR: 1. API内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。
  ```

## aclnnConvolutionBackward

- **参数说明：**

  * workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnConvolutionBackwardGetWorkspaceSize获取。
  * executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制
由于硬件资源限制，算子在部分参数取值组合场景下会执行失败，请根据日志信息提示分析并排查问题。若无法解决，请单击[Link](https://www.hiascend.com/support)获取技术支持。
