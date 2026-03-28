# FastGeluGrad
## 贡献说明
| 贡献者     | 贡献方  | 贡献算子     | 贡献时间      | 贡献内容         |
|---------|------|----------|-----------|--------------|
| enkilee | 社区任务 | FastGelu | 2025/3/22 | 新增FastGelu算子 |

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  `FastGeluGrad‌`是一种在`FastGelu`基础上进行升级的算子，`FastGeluGrad`算子在`FastGelu`的基础上增加了一个输入`dy`，使得其有两个输入和一个输出，不需要进行大幅度的数据搬迁。其主要目的是优化计算过程，减少类型转换的需求。

- 原型信息
  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">FastGeluGrad</th></tr> 
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
    <tr><td rowspan="3" align="center">算子输入</td>
     
    <tr><td align="center">dy</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr>  
    <tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr>  
    
    <tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">z</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr>  
    
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">fast_gelu_grad</td></tr>  
  </table>

## 约束与限制
- dy，x，z，out的数据类型仅支持float32,float16，数据格式只支持ND

## 算子使用
使用该算子前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

### 编译部署
  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}/cann-ops
    ```

  - 执行编译

    ```bash
    bash build.sh
    ```

  - 部署算子包

    ```bash
    bash build_out/CANN-custom_ops-<cann_version>-linux.<arch>.run
    ```

### 运行验证
跳转到对应调用方式目录，参考Readme进行算子运行验证。
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用FastGeluGrad算子。</td>
    </tr>
</table>

