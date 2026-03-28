# aclnnAddLayerNormGrad

## 支持的产品型号

- Atlas 推理系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnAddLayerNormGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAddLayerNormGrad”接口执行计算。
*  `aclnnStatus aclnnAddLayerNormGradGetWorkspaceSize(const aclTensor *dy, const aclTensor *x1, const aclTensor *x2, const aclTensor *rstd, const aclTensor *mean, const aclTensor *gamma, const aclTensor *dsumOptional, const aclTensor *dxOut, const aclTensor *dgammaOut, const aclTensor *dbetaOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
*  `aclnnStatus aclnnAddLayerNormGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述
- **算子功能**：Add与LayerNorm融合算子的反向计算。

- **计算公式**：

  - 正向公式：（D为reduce轴大小）

    $$
    \operatorname{LayerNorm}(x)=\frac{x_i−\operatorname{E}(x)}{\sqrt{\operatorname{Var}(x)+ eps}}*\gamma + \beta
    $$

    $$
    其中\operatorname{E}(x_i)=\frac{1}{D}\sum_{1}^{D}{x_i}
    $$

    $$
    \operatorname{Var}(x_i)=\frac{1}{D}\sum_{1}^{D}{(x_i-\operatorname{E}(x))^2}
    $$

  - 反向公式：

    $$
    \frac{{\rm d}L}{{\rm d}x_i} = \sum_{j}{\frac{{\rm d}L}{{\rm d}y_j} * \gamma_j * \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}}
    $$

    $$
    其中\hat{x_j}=\frac {x_i-\operatorname{E}(x)}{\sqrt{\operatorname{Var}(x)+eps}}
    $$

    $$
    \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}=(\delta_{ij} - \frac{{\rm d}\operatorname{E}(x)}{{\rm d}x_i}) * \frac{1}{\sqrt{\operatorname{Var}(x_i)}}-\frac{1}{\operatorname{Var}(x_i)}(x_j-\operatorname{E}(x))\frac{\rm d \operatorname{Var}(x_i)}{\rm dx}
    $$

    $$
    其中\frac{{\rm d}\operatorname{E}(x)}{{\rm d}x_i}=\frac{1}{D}
    $$

    $$
    \frac{\rm d \operatorname{Var}(x_i)}{\rm dx}=\frac{1}{D}\frac{1}{\sqrt{\operatorname{Var}(x_i)}}(x_i-\operatorname{E}(x))
    $$



## aclnnAddLayerNormGradGetWorkspaceSize

- **参数说明：**
  * dy（aclTensor\*，计算输入）：主要的grad输入，Device侧的aclTensor，shape支持1-8维，数据格式支持ND，不支持非连续输入。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * x1（aclTensor\*，计算输入）：为正向融合算子的输入x1，Device侧的aclTensor，shape需要与dy相同，数据格式支持ND，不支持非连续输入。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * x2（aclTensor\*，计算输入）：为正向融合算子的输入x2，Device侧的aclTensor，shape需要与dy相同，数据格式支持ND，不支持非连续输入。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * rstd（aclTensor\*，计算输入）：表示正向输入x1、x2之和的rstd。输入数据类型支持FLOAT32，shape需要与dy满足broadcast关系（前几维的维度和dy前几维的维度相同，前几维指dy的维度减去gamma的维度，表示不需要norm的维度），数据格式支持ND，不支持非连续输入。不支持空Tensor。
  * mean（aclTensor\*，计算输入）：表示正向输入x1、x2之和的mean。数据类型支持FLOAT32，shape需要与dy满足broadcast关系（前几维的维度和dy前几维的维度相同，前几维指dy的维度减去gamma的维度，表示不需要norm的维度），数据格式支持ND，不支持非连续输入。不支持空Tensor。
  * gamma（aclTensor\*，计算输入）：表示正向输入的gamma，shape维度和dy后几维的维度相同，shape支持1-8维，表示需要norm的维度，数据格式支持ND，不支持非连续输入。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * dsumOptional（aclTensor\*，计算输入）：shape支持2-8维，数据格式支持ND，不支持非连续输入。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * dxOut（aclTensor\*，计算输出）：shape支持2-8维，数据格式支持ND。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * dgammaOut（aclTensor\*，计算输出）：数据类型支持FLOAT32，shape与输入gamma一致，数据格式支持ND。不支持空Tensor。
  * dbetaOut（aclTensor\*，计算输出）：数据类型支持FLOAT32，shape与输入gamma一致，数据格式支持ND。不支持空Tensor。
  * workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  返回161002（ACLNN_ERR_PARAM_INVALID）：输入和输出的数据类型不在支持的范围之内。
  ```


## aclnnAddLayerNormGrad
- **参数说明：**
  * workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAddLayerNormGradGetWorkspaceSize获取。
  * executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制
- **功能维度**
  * 数据类型支持
    * Atlas 推理系列产品：dy、x1、x2、gamma、dsumOptional支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：dy、x1、x2、gamma、dsumOptional支持FLOAT32、FLOAT16、BFLOAT16。
    * rstd、mean支持：FLOAT32。
  * 数据格式支持：ND。
- **未支持类型说明**
  * DOUBLE：指令不支持DOUBLE。
  * 是否支持空tensor：不支持空进空出。
  * 是否非连续tensor：不支持输入非连续，不支持数据非连续。
- **边界值场景说明**
  * 当输入是inf时，输出为inf。
  * 当输入是nan时，输出为nan。
