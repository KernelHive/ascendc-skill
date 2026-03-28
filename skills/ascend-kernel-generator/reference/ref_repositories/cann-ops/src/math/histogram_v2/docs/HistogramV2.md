声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# aclnnHistc

## 支持的产品型号

- Atlas 推理系列产品
- Atlas A2 训练系列产品/Atlas A3 训练系列产品

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnHistcGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnHistc”接口执行计算。

- `aclnnStatus aclnnHistcGetWorkspaceSize(const aclTensor* self, int64_t bins, const aclScalar* min, const aclScalar* max, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnHistc(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

计算张量直方图。
以min和max作为统计上下限，在min和max之前划出等宽的数量为bins的区间，统计张量self中元素在各个区间的数量。如果min和max都为0，则使用张量中所有元素的最小值和最大值作为统计的上下限。小于min和大于max的元素不会被统计。

## aclnnHistcGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入)：待被统计元素在各个bins的数量的张量。Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、INT32、INT64、INT16、INT8、UINT8。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。shape的维度支持0-8维。
  - bins(int64_t, 计算输入)：直方图bins的数量，Host侧的INT64类型，取值范围需大于0。
  - min(aclScalar*, 计算输入)：直方图的统计下限（包括）。Host侧的aclScalar，数据类型需要是可转换成FLOAT的类型，取值范围不能大于max的值。
  - max(aclScalar*, 计算输入)：直方图的统计上限（包括）。Host侧的aclScalar，数据类型需要是可转换成FLOAT的类型，取值范围不能小于min的值。
  - out(aclTensor*, 计算输出)：直方图统计结果。Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、INT32、INT64、INT16、INT8、UINT8，且out数据类型需要可转换为self的数据类型，（参考[互转换关系](common/互转换关系.md)）。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape只支持1维tensor，且元素个数等于bins。
  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、out、min、max是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. self和out的数据类型和数据格式不在支持的范围之内。
                                        2. self与out的数据类型不满足互推导关系。
                                        3. 传入的bins小于0。
                                        4. 传入的min大于max。
                                        5. out的shape维度不为1。
                                        6. self的shape维度大于8。
                                        7. out的size不等于bins。
  ```

## aclnnHistc

- **参数说明：**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnHistcGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制

无。

## 算子原型
<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">HistogramV2</th></tr>
</tr>
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">默认值</td></tr>
<tr><td align="center">x</td><td align="center">-</td><td align="center">float16, float32, int64, int32, int16, int8, uint8</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">min</td><td align="center">-</td><td align="center">float16, float32, int64, int32, int16, int8, uint8</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">max</td><td align="center">-</td><td align="center">float16, float32, int64, int32, int16, int8, uint8</td><td align="center">ND</td><td align="center">\</td></tr>
</tr>

<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">-</td><td align="center">int32</td><td align="center">ND</td><td align="center">\</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">histogram_v2</td></td></tr>
</table>

## 调用示例

详见[HistogramV2自定义算子样例说明算子调用章节](../README.md#算子调用)