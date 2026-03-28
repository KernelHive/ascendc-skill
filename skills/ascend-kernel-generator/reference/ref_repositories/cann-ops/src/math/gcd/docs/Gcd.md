声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# GCD

## 支持的产品型号

Atlas A2 训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：GCD算子计算两个整数张量元素的最大公约数。支持5维张量及广播操作。
- 计算公式：

  $$
  y = gcd(x1, x2)
  $$


## 算子执行接口

* `aclnnStatus aclnnGcdGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnGcd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

### aclnnGcdGetWorkspaceSize

- **参数说明：**
  - x1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，数据类型支持int16,int32,int64，数据格式支持ND。
  - x2（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，数据类型支持int16,int32,int64，数据格式支持ND。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，数据类型支持int16,int32,int64，数据格式支持ND，输出维度与x1一致。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

### aclnnGcd

- **参数说明：**
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*，入参）：op执行器。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

## 约束与限制

1. **输入输出限制**：
   - 输入输出支持5维张量，支持广播操作
   - 输入输出数据类型仅支持int16,int32,int64
   - 输入输出数据格式仅支持ND

2. **性能限制**：
   - 在广播情况下性能受较大影响
   - 对于64位整数输入，计算性能远低于32位或16位整数

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">GCD</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td>
 
<td align="center">x1</td><td align="center">tensor</td><td align="center">int16,int32,int64</td><td align="center">ND</td></tr>  
<td align="center">x2</td><td align="center">tensor</td><td align="center">int16,int32,int64</td><td align="center">ND</td></tr>

<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">int16,int32,int64</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gcd</td></tr>  
</table>
