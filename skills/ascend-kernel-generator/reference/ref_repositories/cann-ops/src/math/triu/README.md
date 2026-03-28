# Triu
## 贡献说明
| 贡献者                                                                    | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|------------------------------------------------------------------------|-----|------|------|------|
| enkilee |  社区任务   |  Triu   |   2025/3/11   |    新增Triu算子  |

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  `Triu`算子用于提取张量的上三角部分。返回一个张量`out`，包含输入矩阵(2D张量)的下三角部分，`out`其余部分被设为0。这里所说的上三角部分为矩阵指定对角线`diagonal`之上的元素。参数`diagonal`控制对角线：默认值是`0`，表示主对角线。如果 `diagonal > 0`，表示主对角线之上的对角线；如果 `diagonal < 0`，表示主对角线之下的对角线。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Triu</th></tr> 
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
    <tr><td rowspan="2" align="center">算子输入</td>
     
    <tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr>  
    
    <tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr>  
    <tr><td rowspan="1" align="center">算子属性</td>
    <td align="center">diagonal</td><td align="center">diagonal</td><td align="center">int</td><td align="center">-</td></tr>  
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">Triu</td></tr>  
  </table>

## 约束与限制
- x,y,out的数据类型仅支持float32,float16，数据格式仅支持ND

## 环境要求
编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Triu算子。</td>
    </tr>
</table>
