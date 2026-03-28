# Cos

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|----|----|----|------|------|
| 韩智惟 | 北京交通大学-赵宏智老师团队 | Cos | 2025/06/25 | 添加Cos算子。|

## 支持的产品型号

- Atlas A2训练系列产品
- Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 算子描述

- 功能描述

  Cos算子提供余弦函数的计算功能。

- 原型信息

  <table>
  <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Cos</th></tr> 
  <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
  <tr><td rowspan="2" align="center">算子输入</td>
  <tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
  <tr><td rowspan="1" align="center">算子输出</td>
  <td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cos</td></tr>  
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