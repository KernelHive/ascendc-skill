# aclnnGridSampler2D

## 支持的产品型号

- Atlas 训练系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnGridSampler2DGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGridSampler2D”接口执行计算。

- `aclnnStatus aclnnGridSampler2DGetWorkspaceSize(const aclTensor *input, const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnGridSampler2D(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：提供一个输入tensor以及一个对应的grid网格，然后根据grid中每个位置提供的坐标信息，将input中对应位置的像素值填充到网格指定的位置，得到最终的输出。
- 计算公式：
  input、grid、output的尺寸如下：

  $$
  input: (N,C,H_{in},W_{in})\\
  grid: (N,H_{out},W_{out},2)\\
  output: (N,C,H_{out},W_{out})
  $$

  其中input、grid、out中的N是一致的，input和output中的C是一致的，grid和output中的H_{out}、W_{out}是一致的，grid最后一维大小为2，表示input像素位置信息为(x,y)，一般会将x和y的取值范围归一化到[-1,1]之间，(-1,1)表示左上角坐标，(1,1)表示右下角坐标。
  - 对于超出范围的坐标，会根据paddingMode进行不同处理：

    - paddingMode=0，表示对越界位置用0填充。
    - padddingMode=1，表示对越界位置用边界值填充。
    - paddingMode=2，表示对越界位置用边界值的对称值填充。

  - 对input采样时，会根据interpolationMode进行不同处理：

    - interpolationMode=0，如果(x,y)没有input对应坐标，则取(x,y)周围四个坐标的加权平均值。
    - interpolationMode=1，表示取input中距离(x,y)最近的坐标值。
    - interpolationMode=2，如果(x,y)没有input对应坐标，则取(x,y)周围十六个坐标的加权平均值。

## aclnnGridSampler2DGetWorkspaceSize

- **参数说明：**

  - input（aclTensor*，计算输入）：Device侧的aclTensor，表示进行采样的输入张量。数据类型支持FLOAT32、FLOAT16、DOUBLE，支持非连续的Tensor，不支持空Tensor。数据格式支持ND。支持shape为$(N,C,H_{in},W_{in})$。H\*W的最大值只支持INT32上界。
  - grid（aclTensor*，计算输入）：Device侧的aclTensor。表示采样点坐标张量。数据类型支持FLOAT32、FLOAT16、DOUBLE。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。支持shape为$(N,H_{out},W_{out},2)$，且N与入参`input`的shape中的N一致。数据类型与入参`input`的数据类型一致。
  - interpolationMode（int64_t，计算输入）：Host侧整型属性，数据类型为int64_t，表示插值模式，0：bilinear（双线性插值），1：nearest（最邻近插值），2：bicubic（双三次插值）。
    - Atlas 训练系列产品：支持的插值模式如下：
      - 0：bilinear（双线性插值）
      - 1：nearest（最邻近插值）
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持的插值模式如下：
      - 0：bilinear（双线性插值）
      - 1：nearest（最邻近插值）
      - 2：bicubic（双三次插值）。仅当input数据类型为FLOAT32或者FLOAT16时支持bicubic插值。
  - paddingMode（int64_t，计算输入）：Host侧整型属性，数据类型为int64_t，表示填充模式，即当（x,y）取值超过输入特征图采样范围时，返回一个特定值，有0：zeros、1：border、2：reflection三种模式。
  - alignCorners（bool，计算输入）：Host侧BOOL类型属性，数据类型为bool，表示设定特征图坐标与特征值的对应方式，设定为true时，特征值位于像素中心。设定为false时，特征值位于像素的角点。
  - out（aclTensor*，计算输出）：Device侧的aclTensor，表示采样后的输出张量。支持shape为$(N,C,H_{out},W_{out})$，且N、C与input的shape中的N、C一致，$H_{out}$、$W_{out}$与grid的shape中的$H_{out}$、$W_{out}$一致。数据类型支持FLOAT32、FLOAT16、DOUBLE，且数据类型与input的数据类型一致。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的input、grid或out是空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. input、grid、out的数据类型不在支持的范围之内或数据类型不一致。
                                      2. input、grid、out的数据格式不在支持的范围之内。
                                      3. interpolationMode或paddingMode的值不在支持范围内。
                                      4. interpolationMode为bicubic时，input、grid、out的数据类型不是FLOAT32或者FLOAT16。
                                      5. input、grid、out的维度关系不匹配。
                                      6. input最后两维为空。
  ```

## aclnnGridSampler2D

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnGridSampler2DGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制
- input的shape，后两维不能为0。
- grid的输入值*图片（长或宽）大于24位的二进制数（16777216），采样点可能存在误差，精度可能产生偏差。
- 如果grid含有大量超过[-1,1]范围的数据，使用zeros或者border的填充策略时，计算结果中的值会大量重复。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：
  - 如果grid存在超出[-1,1]范围的数据，使用bicubic插值时，小值域数据计算可能存在误差，精度可能产生偏差。
  - 使用bilinear或者bicubic插值时，针对FLOAT16数据类型，需要使用workspace内存。
- Atlas 训练系列产品：使用bilinear差值时，针对FLOAT16数据类型，需要使用workspace内存。


