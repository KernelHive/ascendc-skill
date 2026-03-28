# ScatterReduce
## 贡献说明
| 贡献者        | 贡献方 | 贡献算子          | 贡献时间      | 贡献内容              |
|------------|-----|---------------|-----------|-------------------|
| zzh-stable | 算子赛 | ScatterReduce | 2025/6/19 | 新增ScatterReduce算子 |

## 支持的产品型号
- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  `ScatterReduce`算子将源张量（`src`）的值按照索引（`index`）规则归约到目标张量（`x`）的指定维度（`dim`），支持多种归约操作。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">ScatterReduce</th></tr>  
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
    <tr><td rowspan="3" align="center">算子输入</td>
        <td align="center">x</td><td align="center">tensor</td><td align="center">fp32, fp16</td><td align="center">ND</td></tr>  
        <td align="center">index</td><td align="center">tensor</td><td align="center">int32</td><td align="center">ND</td></tr>  
        <td align="center">src</td><td align="center">tensor</td><td align="center">fp32, fp16</td><td align="center">ND</td></tr>  
    <tr><td rowspan="1" align="center">算子输出</td>
        <td align="center">y</td><td align="center">tensor</td><td align="center">fp32, fp16</td><td align="center">ND</td></tr>  
    <tr><td rowspan="3" align="center">attr属性</td>
        <td align="center">dim</td><td align="center">int</td><td colspan="2" align="center">required</td></tr>
        <td align="center">reduce</td><td align="center">str</td><td colspan="2" align="center">required</td></tr>
        <td align="center">include_self</td><td align="center">bool</td><td colspan="2" align="center">default: TRUE</td></tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">scatter_reduce</td></tr>  
  </table>

## 约束与限制
- x,index,src,y的数据类型仅支持fp32, fp16，int32，数据格式仅支持ND

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用ScatterReduce算子。</td>
    </tr>
</table>
