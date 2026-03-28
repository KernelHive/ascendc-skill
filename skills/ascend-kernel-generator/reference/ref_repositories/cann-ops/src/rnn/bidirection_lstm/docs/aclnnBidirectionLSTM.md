# aclnnBidirectionLSTM

## 支持的产品型号
- Atlas 推理系列加速卡产品

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnBidirectionLSTMGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBidirectionLSTM”接口执行计算。

- `aclnnStatus aclnnBidirectionLSTMGetWorkspaceSize(const aclTensor *x, const aclTensor *initH, const aclTensor *initC, const aclTensor *wIh, const aclTensor *wHh, const aclTensor *bIhOptional, const aclTensor *bHhOptional, const aclTensor *wIhReverseOptional, const aclTensor *wHhReverseOptional, const aclTensor *bIhReverseOptional, const aclTensor *bHhReverseOptional, int64_t numLayers, bool isbias, bool batchFirst, bool bidirection, const aclTensor *yOut, const aclTensor *outputHOut, const aclTensor *outputCOut, uint64_t *workspaceSize, aclOpExecutor **executor);`
- `aclnnStatus aclnnBidirectionLSTM(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);`

## 功能描述

- 算子功能：LSTM（Long Short-Term Memory，长短时记忆）网络是一种特殊的循环神经网络（RNN）模型。进行LSTM网络计算，接收输入序列和初始状态，返回输出序列和最终状态。
- 计算公式：
  
  $$
  f_t =sigm(W_f[h_{t-1}, x_t] + b_f)\\
  i_t =sigm(W_i[h_{t-1}, x_t] + b_i)\\
  o_t =sigm(W_o[h_{t-1}, x_t] + b_o)\\
  \tilde{c}_t =tanh(W_c[h_{t-1}, x_t] + b_c)\\
  c_t =f_t ⊙ c_{t-1} + i_t ⊙ \tilde{c}_t\\
  c_{o}^{t} =tanh(c_t)\\
  h_t =o_t ⊙ c_{o}^{t}\\
  $$

  - $x_t ∈ R^{d}$：LSTM单元的输入向量。
  - $f_t ∈ (0, 1)^{h}$：遗忘门激活向量。
  - $i_t ∈ (0, 1)^{h}$：输入门、更新门激活向量。
  - $o_t ∈ (0, 1)^{h}$：输出门激活向量。
  - $h_i ∈ (-1, 1)^{h}$：隐藏状态向量，也称为LSTM单元的输出向量。
  - $\tilde{c}_t ∈ (-1, 1)^{h}$：cell输入激活向量。
  - $c_t ∈ R^{h}$：cell状态向量。
  - $W ∈ R^{h×d}，(U ∈ R^{h×h})∩(b ∈ R^{h})$：训练中需要学习的权重矩阵和偏置向量参数。

## aclnnBidirectionLSTMGetWorkspaceSize

- **参数说明：**
  - x（aclTensor\*，计算输入）：Device侧的aclTensor，LSTM单元的输入向量，数据类型支持FLOAT16。只支持连续Tensor，数据格式支持ND。shape支持三维（time_step, batch_size, input_size）。其中，`time_step`表示时间维度；`batch_size`表示每个时刻需要处理的batch数量；`input_size`表示输入的特征数量。
  - initH（aclTensor\*，计算输入）：Device侧的aclTensor，初始化hidden状态，数据类型支持FLOAT16。只支持连续Tensor，数据格式支持ND。shape支持三维（num_layers, batch_size, hidden_size）或者当bidirection为True时（2 * num_layers, batch_size, hidden_size）。其中，`num_layers`对应参数`numLayers`，表示LSTM层数；`hidden_size`表示隐藏状态的特征数量。
  - initC（aclTensor\*，计算输入）：Device侧的aclTensor，初始化cell状态，数据类型支持FLOAT16。只支持连续Tensor，数据格式支持ND。shape支持三维（num_layers, batch_size, hidden_size）或者当bidirection为True时（2 * num_layers, batch_size, hidden_size）。
  - wIh（aclTensor\*，计算输入）：Device侧的aclTensor，input-hidden权重，数据类型支持FLOAT16。只支持连续Tensor，数据格式支持ND。shape支持二维（4 * hidden_size, input_size）。
  - wHh（aclTensor\*，计算输入）：Device侧的aclTensor，hidden-hidden权重，数据类型支持FLOAT16。只支持连续Tensor，数据格式支持ND。shape支持二维（4 * hidden_size, hidden_size）。
  - bIhOptional（aclTensor\*，计算输入）：Device侧的aclTensor，input-hidden偏移，数据类型支持FLOAT16。只支持连续Tensor，数据格式支持ND。shape支持一维（4 * hidden_size）。
  - bHhOptional（aclTensor\*，计算输入）：Device侧的aclTensor，hidden-hidden偏移，数据类型支持FLOAT16。只支持连续Tensor，数据格式支持ND。shape支持一维（4 * hidden_size）。
  - wIhReverseOptional（aclTensor\*，计算输入）：Device侧的aclTensor，逆向input-hidden权重，数据类型支持FLOAT16。只支持连续Tensor，数据格式支持ND。shape支持二维（4 * hidden_size, input_size）。
  - wHhReverseOptional（aclTensor\*，计算输入）：Device侧的aclTensor，逆向hidden-hidden权重，数据类型支持FLOAT16。只支持连续Tensor，数据格式支持ND。shape支持二维（4 * hidden_size, input_size）。
  - bIhReverseOptional（aclTensor\*，计算输入）：Device侧的aclTensor，逆向input-hidden偏移，数据类型支持FLOAT16。只支持连续Tensor，数据格式支持ND。shape支持一维（4 * hidden_size）。
  - bHhReverseOptional（aclTensor\*，计算输入）：Device侧的aclTensor，逆向hidden-hidden偏移，数据类型支持FLOAT16。只支持连续Tensor，数据格式支持ND。shape支持一维（4 * hidden_size）。
  - numLayers（int64\_t，计算输入）：Host侧的整型，表示LSTM层数，当前只支持1，类型支持int64\_t。
  - isbias（bool，计算输入）：Host侧的bool，表示是否有bias，类型支持bool。
  - batchFirst（bool，计算输入）：Host侧的bool，表示batch是否是第一维，当前只支持false，类型支持bool。
  - bidirection（bool，计算输入）：Host侧的bool，表示是否是双向，类型支持bool。
  - yOut（aclTensor\*，计算输出）：Device侧的aclTensor，LSTM单元的输出向量，数据类型支持FLOAT16。只支持连续Tensor，数据格式支持ND。shape支持三维（time_step, batch_size, hidden_size）或者当bidirection为True时（time_step, batch_size, 2 * hidden_size）。
  - outputHOut（aclTensor\*，计算输出）：Device侧的aclTensor，最终hidden状态，数据类型支持FLOAT16，只支持连续Tensor，数据格式支持ND。shape支持三维（num_layers, batch_size, hidden_size）或者当bidirection为True时（2 * num_layers, batch_size, hidden_size）。
  - outputCOut（aclTensor\*，计算输出）：Device侧的aclTensor，最终cell状态，数据类型支持FLOAT16，只支持连续Tensor，数据格式支持ND。shape支持三维（num_layers, batch_size, hidden_size）或者当bidirection为True时（2 * num_layers, batch_size, hidden_size）。
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：如果传入参数类型为aclTensor且其数据类型不在支持的范围之内。
  返回561002（ACLNN_ERR_INNER_TILING_ERROR）：如果传入参数类型为aclTensor且其shape与上述参数说明不符。
  ```

## aclnnBidirectionLSTM

- **参数说明：**
  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnBidirectionLSTMGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream ，入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus: 返回状态码。

## 约束与限制

无。
