声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# MatmulApiConstant

## 支持的产品型号

Atlas A2训练系列产品/Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：在调用Matmul API实现矩阵乘法时，通过使用全量常量化的MatmulApiStaticTiling模板参数，替代非常量的TCubeTiling参数。将计算提前到编译期，以减少运行时的Scalar计算开销，实现算子性能提升。
适用场景：
  - Matmul初始化时的Scalar计算较多，影响指令头开销。
  - Matmul迭代之间的Scalar计算较多，阻塞MTE2流水。
限制：需要在编译期就确定部分Tiling参数，需满足以下条件之一：
  - 全量常量化：可以确定的Single Shape（SingleCoreM/SingleCoreN/SingelCoreK）和Base Shape（baseM/baseN/baseK）。
  - 部分常量化：可以确定的Base Shape（baseM/baseN/baseK）。
其中全量常量化场景比部分常量化场景减少更多的Scalar计算开销。
- 计算公式：
  
  $$
  c = (a × b) + bias
  $$
  
  **说明：**
  无。

## 实现原理

`MatmulApiConstant`算子使用`Ascend C`的高阶API`Matmul`接口，并使用全量常量化的MatmulApiStaticTiling模板参数，替代非常量的TCubeTiling参数。将计算提前到编译期，以减少运行时的Scalar计算开销，实现Matmul算子性能提升。

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnMatmulApiConstantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMatmulApiConstant”接口执行计算。

* `aclnnStatus aclnnMatmulApiConstantGetWorkspaceSize(const aclTensor *a, const aclTensor *b, const aclTensor *bias, const aclTensor *c, uint64_t workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMatmulApiConstant(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnMatmulApiConstantGetWorkspaceSize

- **参数说明：**
  
  - a（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入a，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - b（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入b，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - bias（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入bias，数据类型支持FLOAT，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。  
  - c（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出c，数据类型支持FLOAT，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：a、b、bias, c的数据类型和数据格式不在支持的范围内。
  ```

### aclnnMatmulApiConstant

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMatmulApiConstantGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- a，b的数据类型只支持FLOAT16，bias, c的数据类型只支持FLOAT, 数据格式只支持ND

## 算子原型

算子MatmulApiConstant注册的原型如下，支持的shape为：M = 1024, N = 640, K = 256。
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">MatmulApiConstant</td></tr>
</tr>
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">bias</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_api_constant</td></tr>
</table>

## 调用示例

详见[MatmulApiConstant自定义算子样例说明算子调用章节](../README.md#算子调用)
