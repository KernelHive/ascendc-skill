声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# GroupNormSilu

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2推理产品
- Atlas A3 训练系列产品/Atlas 800I A3推理产品
- Atlas 200I/500 A2推理产品
- Atlas 推理系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 接口功能：计算输入self的组归一化结果out，均值meanOut，标准差的倒数rstdOut，以及silu的输出。

- 计算公式：
  记 $x=self$, $E[x] = \bar{x}$代表$x$的均值，$Var[x] = \frac{1}{n - 1} * \sum_{i=1}^n(x_i - E[x])^2$代表$x$的样本方差，则
  $$
  \left\{
  \begin{array} {rcl}
  out& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
  meanOut& &= E[x]\\
  rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
  \end{array}
  \right.
  $$

## 算子执行接口

### aclnnGroupNormSiluGetWorkspaceSize
每个算子分为两段式接口，必须先调用 “aclnnGroupNormSiluGetWorkspaceSize” 接口获取入参并根据计算流程计算所需workspace大小以及包含了算子计算流程的执行器，再调用 “aclnnGroupNormSilu” 接口执行计算。

* `aclnnStatus aclnnGroupNormSiluGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta, int64_t group, double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnGroupNormSilu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

### aclnnGroupNormSiluV2GetWorkspaceSize
每个算子分为两段式接口，必须先调用 “aclnnGroupNormSiluV2GetWorkspaceSize” 接口获取入参并根据计算流程计算所需workspace大小以及包含了算子计算流程的执行器，再调用 “aclnnGroupNormSiluV2” 接口执行计算。

* `aclnnStatus aclnnGroupNormSiluV2GetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta, int64_t group, double eps, bool activateSilu, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnGroupNormSiluV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`


**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。

## aclnnGroupNormSiluGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入)：`out`计算公式中的$x$，数据类型支持BFLOAT16、FLOAT16、FLOAT，维度需大于一维，数据格式支持ND，支持非连续的Tensor。
  - gamma(aclTensor*, 计算输入):可选参数，`out`计算公式中的$\gamma$，数据类型支持BFLOAT16、FLOAT16、FLOAT，维度为一维，元素数量需与输入$self$的第1维度保持相同，数据格式支持ND，支持非连续的Tensor。
  - beta(aclTensor*, 计算输入):可选参数，`out`计算公式中的$\beta$，数据类型支持BFLOAT16、FLOAT16、FLOAT，维度为一维，元素数量需与输入$self$的第1维度保持相同，数据格式支持ND，支持非连续的Tensor。
  - group(int, 计算输入): INT32或者INT64常量，表示将输入$self$的第1维度分为group组。
  - eps(double, 计算输入): DOUBLE常量，`out`和`rstdOut`计算公式中的$eps$值。
  - out(aclTensor*, 计算输出): 输出张量，数据类型支持BFLOAT16、FLOAT16、FLOAT，数据类型和shape与$self$相同，数据格式支持ND，支持非连续的Tensor。
  - meanOut(aclTensor*, 计算输出): 输出张量，数据类型支持BFLOAT16、FLOAT16、FLOAT，数据类型与$self$相同，shape为`(N, group)`，其中`N`与$self$的第0维度保持一致，数据格式支持ND，支持非连续的Tensor。
  - rstdOut(aclTensor*, 计算输出): 输出张量，数据类型支持BFLOAT16、FLOAT16、FLOAT，数据类型与$self$相同，shape为`(N, group)`，其中`N`与$self$的第0维度保持一致，数据格式支持ND，支持非连续的Tensor。
  - workspaceSize(uint64_t\*, 出参): 返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor **, 出参): 返回op执行器，包含算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 ACLNN_ERR_PARAM_NULLPTR：1. 传入的self、out、meanOut、rstdOut是空指针时。
  161002 ACLNN_ERR_PARAM_INVALID：1. self、gamma、beta、out、meanOut、rstdOut数据类型不在支持的范围之内。
                                  2. out、meanOut、rstdOut的数据类型与self相同，gamma、beta与self可以不同。
                                  3. gamma与beta的数据类型必须保持一致，且数据类型与self相同或者为FLOAT。
                                  4. self维度不大于1。
                                  5. self第1维度不能被group整除
                                  6. eps小于等于0。
                                  7. out的shape与self不同。
                                  8. meanOut与rstdOut的shape不为(N, group)，其中N为self第0维度值。
                                  9. gamma不为1维或元素数量不等于输入self第1维度。
                                  10. beta不为1维或元素数量不等于输入self第1维度。
                                  11. group小于等于0。
                                  12. self第0维小于等于0.
                                  13. self第1维小于等于0.
  ```

## aclnnGroupNormSilu

- **参数说明：**
  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnGroupNormSiluGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## aclnnGroupNormSiluV2GetWorkspaceSize

- **参数说明：**

  * self(aclTensor*, 计算输入)：`out`计算公式中的$x$，数据类型支持BFLOAT16、FLOAT16、FLOAT，维度需大于一维，数据格式支持ND，支持非连续的Tensor。
  * gamma(aclTensor*, 计算输入):可选参数，`out`计算公式中的$\gamma$，数据类型支持BFLOAT16、FLOAT16、FLOAT，维度为一维，元素数量需与输入$self$的第1维度相同，数据格式支持ND，支持非连续的Tensor。
  * beta(aclTensor*, 计算输入):可选参数，`out`计算公式中的$\beta$，数据类型支持BFLOAT16、FLOAT16、FLOAT，维度为一维，元素数量需与输入$self$的第1维度相同，数据格式支持ND，支持非连续的Tensor。
  * group(int, 计算输入): INT32或者INT64常量，表示将输入$self$的第1维度分为group组。
  * eps(double, 计算输入): DOUBLE常量，`out`和`rstdOut`计算公式中的$eps$值。
  * activateSilu(bool, 计算输入): BOOL常量，是否支持silu计算。如果设置为true，则表示groupnorm计算后继续silu计算。
  * out(aclTensor*, 计算输出): 输出张量，数据类型支持BFLOAT16、FLOAT16、FLOAT，数据类型和shape与$self$相同，数据格式支持ND，支持非连续的Tensor。
  * meanOut(aclTensor*, 计算输出): 输出张量，数据类型支持BFLOAT16、FLOAT16、FLOAT，数据类型与$self$相同，shape为`(N, group)`，其中`N`与$self$的第0维度保持一致，数据格式支持ND，支持非连续的Tensor。
  * rstdOut(aclTensor*, 计算输出): 输出张量，数据类型支持BFLOAT16、FLOAT16、FLOAT，数据类型与$self$相同，shape为`(N, group)`，其中`N`与$self$的第0维度保持一致，数据格式支持ND，支持非连续的Tensor。
  * workspaceSize(uint64_t\*, 出参): 返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor **, 出参): 返回op执行器，包含算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
161001 ACLNN_ERR_PARAM_NULLPTR：1. 传入的self、out、meanOut、rstdOut是空指针时。
161002 ACLNN_ERR_PARAM_INVALID：1. self、gamma、beta、out、meanOut、rstdOut数据类型不在支持的范围之内。
                                2. out、meanOut、rstdOut的数据类型与self相同，gamma、beta与self可以不同。
                                3. gamma与beta的数据类型必须保持一致，且数据类型与self相同或者为FLOAT。
                                4. self维度不大于1。
                                5. self第1维度不能被group整除
                                6. eps小于等于0。
                                7. out的shape与self不同。
                                8. meanOut与rstdOut的shape不为(N, group)，其中N为self第0维度值。
                                9. gamma不为1维或元素数量不等于输入self第1维度。
                                10. beta不为1维或元素数量不等于输入self第1维度。
                                11. group小于等于0。
                                12. self第0维小于等于0.
                                13. self第1维小于等于0.
```

## aclnnGroupNormSiluV2

- **参数说明：**

  * workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnGroupNormSiluV2GetWorkspaceSize获取。
  * executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- 无。

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">GroupNormSilu</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td> 
<tr><td align="center">self</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入</td> 
<tr><td align="center">gamma</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">beta</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输出</td>
<tr><td align="center">out</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输出</td>
<tr><td align="center">meanOut</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输出</td>
<tr><td align="center">rstdOut</td><td align="center">tensor</td><td align="center">int64</td><td align="center">ND</td></tr>  

<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">group</td><td align="center">scalar</td><td align="center">int64</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">eps</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">activateSilu</td><td align="center">scalar</td><td align="center">bool</td><td align="center">-</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">group_norm_silu</td></tr>  
</table>

## 调用示例

详见[GroupNormSilu自定义算子样例说明算子调用章节](../README.md#算子调用)