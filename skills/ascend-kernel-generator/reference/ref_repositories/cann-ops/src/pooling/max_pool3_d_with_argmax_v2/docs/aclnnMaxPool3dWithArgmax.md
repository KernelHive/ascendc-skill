# aclnnMaxPool3dWithArgmax

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnMaxPool3dWithArgmaxGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMaxPool3dWithArgmax”接口执行计算。

* `aclnnStatus aclnnMaxPool3dWithArgmaxGetWorkspaceSize(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor);`
* `aclnnStatus aclnnMaxPool3dWithArgmax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);`

## 功能描述

* 算子功能：
  * 对于输入信号的输入通道，提供3维最大池化（Max pooling）操作，输出池化后的值out和索引indices。
  * 输入dims的描述：N - 批次，C - 通道，D - 深度，W - 宽度，H - 高度。
  * 当D * H * W超过int32时，建议在模型尺寸上分割D轴。
* 计算公式：
  
  * output tensor中每个元素的计算公式：
    
    $$
    out(N_i, C_j, d, h, w) = \max\limits_{{k\in[0,k_{D}-1],m\in[0,k_{H}-1],n\in[0,k_{W}-1]}}input(N_i,C_j,stride[0]\times d + k, stride[1]\times h + m, stride[2]\times w + n)
    $$
  * out tensor的shape推导公式 (默认ceilMode=false，即向下取整)：
    
    $$
    [N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lfloor{\frac{D_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rfloor + 1,\lfloor{\frac{H_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rfloor + 1, \lfloor{\frac{W_{in}+2 \times {padding[2] - dilation[2] \times(kernelSize[2] - 1) - 1}}{stride[2]}}\rfloor + 1]
    $$
  * out tensor的shape推导公式 (默认ceilMode=true，即向上取整)：
    
    $$
    [N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lceil{\frac{D_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rceil + 1,\lceil{\frac{H_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rceil + 1, \lceil{\frac{W_{in}+2 \times {padding[2] - dilation[2] \times(kernelSize[2] - 1) - 1}}{stride[2]}}\rceil + 1]
    $$

## aclnnMaxPool3dWithArgmaxGetWorkSpaceSize

* **参数说明**：
  
  * self(aclTensor*, 计算输入): 输入Tensor，Device侧aclTensor。数据类型仅支持FLOAT32、FLOAT16、BFLOAT16。shape支持4D、5D。支持非连续的Tensor，数据格式支持ND。
  * kernelSize(aclIntArray*, 计算输入): 表示最大池化的窗口大小，数组长度必须为1或3，且数组元素必须都大于0。
  * stride(aclIntArray*, 计算输入): 窗口移动的步长，数组长度必须为0，1或3，且数组元素必须都大于0。当数组的长度为0时，内部会取kernelSize的值作为strides。
  * padding(aclIntArray*, 计算输入): 每一条边补充的层数，补充的位置填写“负无穷”。数组长度必须为1或3，且数组元素必须都大于等于0且小于等于kernelSize/2。
  * dilation(aclIntArray*, 计算输入): 控制窗口中元素的步幅，数组长度必须为1或3，值仅支持1。
  * ceilMode(bool, 计算输入): 为True时表示计算输出形状时用向上取整的方法，为False时则表示向下取整。
  * out(aclTensor \*, 计算输出): 输出Tensor，是Device侧aclTensor。池化后的结果。数据类型仅支持FLOAT32、FLOAT16、BFLOAT16和self保持一致。shape由上述公式推导出。数据格式支持ND，与self保持一致。
  * indices(aclTensor \*, 计算输出): 输出Tensor，是Device侧aclTensor。最大值的索引位置组成的Tensor。数据类型仅支持INT32。shape和out保持一致。数据格式aceSize(uint64_t \*, 出参): 返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor \*\*, 出参): 返回op执行器，包含了算子计算流程。
* **返回值**：
  
  aclnnStatus: 返回状态码。

```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self、out是空指针。
  161002(ACLNN_ERR_PARAM_INVALID)：1. self的数据类型不在支持的范围内。
                                   2. self的数据格式不在支持的范围内。
                                   3. self的shape不是4维, 5维。
                                   4. 通过公式推导出的out的shape的某个轴为0。
                                   5. kernelSize中的数值中存在小于等于0的数值。
                                   6. kernelSize的长度不等于1或3;
                                   7. stride的数值中存在小于等于0的值。
                                   8. stride的长度不等于0, 1或3(stride长度为0时，stride的数值等于kernelSize的值);
                                   9. padding的数值中存在小于0或者大于kernelSize/2的值。
                                   10. padding的长度不等于1或3;
                                   11. dilation中的数值中存在不等于1的数值
                                   12. 平台不支持
                                   13. depth * height * width > max int32, 超出了Indices的表达范围


  561103（ACLNN_ERR_INNER_NULLPTR）: 1. 中间结果为null。
  561101（ACLNN_ERR_INNER_CREATE_EXECUTOR）: 1. 执行者为null。
```

## aclnnMaxPool3dWithArgmax

- **参数说明：**
  
  * workspace(void \*, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnMaxPool3dWithArgmaxGetWorkSpaceSize获取。
  * executor(aclOpExecutor \*, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的 AscendCL Stream流。
- **返回值：**
  
  aclnnStatus: 返回状态码，具体参见aclnn返回码。

## 约束与限制

- 输入tensor的数据类型仅支持FLOAT32、FLOAT16、BFLOAT16。
- 输入数据排布不支持NDHWC。
- kernelSize、stride、padding、dilation、ceilMode参数需要保证输出out shape中不存在小于1的轴。
- 当ceilMode为True的时候，如果滑动窗口全部在右侧padding区域上，这个输出结果将被忽略。