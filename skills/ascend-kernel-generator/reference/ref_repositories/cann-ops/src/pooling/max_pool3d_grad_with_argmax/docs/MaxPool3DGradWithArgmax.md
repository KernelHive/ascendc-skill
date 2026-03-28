# MaxPool3DGradWithArgmax

## 支持的产品型号

 Atlas A2 训练系列产品/Atlas 800I A2 推理产品

## 功能描述

- 算子功能：
正向最大池化[aclnnMaxPool3dWithArgmax]的反向传播，将梯度回填到每个窗口最大值的坐标处，相同坐标处累加。

## aclnnMaxPool3dWithArgmaxBackwardGetWorkSpaceSize

- **参数说明：**
  * gradOutput(aclTensor*, 计算输入): 梯度Tensor，Device侧aclTensor。和正向的输出shape一致。支持非连续的Tensor，数据格式支持ND, 当输入是5维时，内部按照NCDHW处理，当输入是4维时，在0维度处补1，内部按照NCDHW处理。
    * 昇腾910B AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * self(aclTensor*, 计算输入): 正向的输入Tensor，Device侧aclTensor。支持非连续的Tensor，数据格式支持ND, 当输入是5维时，内部按照NCDHW处理，当输入是4维时，在0维度处补1，内部按照NCDHW处理，与gradOutput一致。
    * 昇腾910B AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * indices(aclTensor \*, 计算输入): 输入Tensor，是Device侧aclTensor。正向输入中最大元素的索引位置。数据格式支持NCDHW，与self保持一致。shape与gradOutput一致。
    * 昇腾910B AI处理器：数据类型仅支持INT32
  * kernelSize(aclIntArray*, 计算输入): 表示最大池化的窗口大小。Host侧的aclIntArray，表示池化窗口的大小，INT64类型数组，长度为1 ($kD = kH = kW$) 或3 ($kD, kH, kW$)。
  * stride(aclIntArray*, 计算输入): Host侧的aclIntArray，表示池化操作的步长，INT64类型的数组，长度为0（$sD = kD, sH = kH, sW = kW$）或者1（$sD = sH = sW$）或3（$sD, sH, sW$）。
  * padding(aclIntArray*, 计算输入): Host侧的aclIntArray，表示在输入的D、H、W方向上padding补0的层数，INT64类型数组，长度为1（$padD = padH = padW$）或3（$padD, padH, padW$）。
  * dilation(aclIntArray*, 计算输入): Host侧的aclIntArray，表示控制窗口中元素的步幅，INT64类型数组，长度为1（$dD = dH = dW$）或3（$dD, dH, dW$），值仅支持1。
  * ceilMode(const bool \*, 计算输入): 表示正向平均池化过程中推导的输出的shape是否向上取整。数据类型支持BOOL。
  * gradInput(aclTensor \*, 计算输出): 反向输出Tensor，是Device侧aclTensor。shape与self保持一致。支持NCDHW，与self保持一致。
    * 昇腾910B AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * workspaceSize(uint64_t \*, 出参): 返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor \*\*, 出参): 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self、indices是空指针。
  161002(ACLNN_ERR_PARAM_INVALID)：1. gradOutput、self、indices、gradInput的数据类型不在支持的范围内。
                                   2. gradOutput、self、indices、gradInput的数据格式不在支持的范围内。
                                   3. gradOutput与indices的shape不一致，self和gradInput的shape不一致。
                                   4. kernelSize的长度不等于1或者3。
                                   5. kernelSize中的数值中存在小于等于0的数值。
                                   6. stride的长度不等于0，1或3。
                                   7. stride的数值中存在小于等于0的值。
                                   8. padding的长度不等于1或3.
                                   9. padding的数值中存在小于0或者大于kernelSize/2的值。
                                   10. dilation的数值不等于1。
                                   11. 平台不支持
                                   12. depth * height * width > max int32，超出了indices的表达范围。
  ```

## aclnnMaxPool3dWithArgmaxBackward

- **参数说明：**
  * workspace(void \*, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnMaxPool3dWithArgmaxBackwardGetWorkSpaceSize获取。
  * executor(aclOpExecutor \*, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus: 返回状态码。

## 约束与限制
- **功能维度**
  * 数据类型支持
    * indices支持INT32。
  * 数据格式支持：ND。
- **未支持类型说明**
  * DOUBLE：指令不支持DOUBLE。
  * 是否支持空tensor：不支持空进空出。
- **边界值场景说明**
  * 当输入是inf时，输出为inf。
  * 当输入是nan时，输出为nan。