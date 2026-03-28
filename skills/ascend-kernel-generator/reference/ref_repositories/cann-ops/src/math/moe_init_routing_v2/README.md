# MoeInitRoutingV2
## 贡献说明
| 贡献者       | 贡献方                    | 贡献算子        | 贡献时间 | 贡献内容 |
|-----------|------------------------|-------------|------|------|
| chenmohua | IFLYTEK BITBRAIN（科大讯飞） | MoeInitRoutingV2 |  2025/3/26  |   新增MoeInitRoutingV2算子   |

## 支持的产品型号

- Atlas A2训练系列产品
- Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  `Moe_init_routing_v2`算子在aclnnMoeInitRoutingV2的基础上增加了对expandedX 和 expandRowId按EP规则进行切分。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Moe_init_routing_v2</th></tr> 
    <tr><td align="center"> </td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>  
    <tr><td rowspan="4" align="center">新增算子输入</td>
    <tr><td align="center">start_expertId</td><td align="center">1</td><td align="center">int32</td><td align="center">-</td></tr>  
    
    
    <tr><td align="center">end_expertId</td><td align="center">1</td><td align="center">int32</td><td align="center">-</td></tr> 
    
    <tr><td align="center">device_id</td><td align="center">1</td><td align="center">int32</td><td align="center">-</td></tr> 
    
    <tr><td rowspan="2" align="center">算子输出变更</td>
    <td align="center">localexpandedXOut</td><td align="center">NUM_ROWS * K, H</td><td align="center">float 16</td><td align="center">ND</td></tr> 
    <td align="center">localexpandedRowIdxOut</td><td align="center">NUM_ROWS * K, </td><td align="center">int32</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">Moe_init_routing_v2</td></tr>  
  </table>

## 约束与限制

- 仅支持dropPadMode=0，expertTokensCountOrCumsumFlag场景下EP规则切分。 

## 算子使用

使用此算子前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Moe_init_routing_v2算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/ATBInvocation"> ATBInvocation</td><td>通过ATB调用的方式调用Moe_init_routing_v2算子。</td>
    </tr>
</table>
 