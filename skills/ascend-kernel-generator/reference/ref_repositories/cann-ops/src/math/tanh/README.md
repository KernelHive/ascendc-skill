# Tanh

## 贡献说明

| 贡献者 | 贡献方                      | 贡献算子 | 贡献时间   | 贡献内容                                   |
| :----: | --------------------------- | -------- | ---------- | ------------------------------------------ |
| 陈子豪 | 浙江工业大学-智能计算研究所 | Tanh     | 2025/06/24 | 新增Tanh算子，实现了双曲正切函数计算功能。 |

## 支持的产品型号

- Atlas A2训练系列产品    

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 算子描述

- 功能描述    

  `Tanh`算子对输入张量的每个元素进行双曲正切函数计算，输出一个与输入形状相同的张量。`Tanh` 将输入值映射到 (-1, 1) 区间，具有“S”形曲线特性，常用于神经网络中的激活函数。

- 原型信息    

  <table>
  <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Tanh</th></tr> 
  <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
  <tr><td rowspan="2" align="center">算子输入</td>
  <tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
  <tr><td rowspan="1" align="center">算子输出</td>
  <td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">tanh</td></tr>  
  </table>

## 约束与限制

- x，out的数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式只支持ND

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
    <th>调用方式</th><th>链接</th>
    <tr>
        <td>aclnn单算子调用</td><td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td>
    </tr>
</table>