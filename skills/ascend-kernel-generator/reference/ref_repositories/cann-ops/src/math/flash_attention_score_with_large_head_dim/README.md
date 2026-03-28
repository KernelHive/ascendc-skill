# FlashAttentionScore

## 贡献说明
| 贡献者    | 贡献方  | 贡献算子                | 贡献时间      | 贡献内容                    |
|--------|------|---------------------|-----------|-------------------------|
| ythqwq | 面壁智能 | FlashAttentionScore | 2025/3/25 | 新增FlashAttentionScore算子 |


## 支持的产品型号

- Atlas A2训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)


## 算子描述
- 功能描述

  训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算的功能，该算子实现了S2>1024条件下的算子功能，完成了该场景下算子的泛化实现。

- 原型信息

  <table>
    <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">FlashAttentionWithLargeHeadDim</td></tr>
    </tr>
    <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td align="center">query</td><td align="center">B,S1,H1</td><td align="center">float16</td><td align="center">ND</td></tr>
    <tr><td align="center">key</td><td align="center">B,S2,H2(S2>1024)</td><td align="center">float16</td><td align="center">ND</td></tr>
    <tr><td align="center">value</td><td align="center">B,S2,H2(S2>1024)</td><td align="center">float16</td><td align="center">ND</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="3" align="center">算子输出</td><td align="center">softmax_max</td><td align="center">B,1,S1,8</td><td align="center">float32</td><td align="center">ND</td></tr>
    <td align="center">softmax_sum</td><td align="center">B,1,S1,8</td><td align="center">float32</td><td align="center">ND</td></tr>
    <td align="center">attention_out</td><td align="center">B,S1,H1</td><td align="center">float16</td><td align="center">ND</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="3" align="center">算子属性</td><td align="center">scale_value</td><td align="center">-</td><td align="center">float32</td><td align="center">ND</td></tr>
    <td align="center">head_num</td><td align="center">-</td><td align="center">int</td><td align="center">ND</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">flash_attention_score_with_large_head_dim</td></tr>
  </table>

## 约束与限制

- query，key，value，softmax_sum，attention_out的数据类型仅支持float32，float16，数据格式仅支持ND

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用FlashAttentionScore算子。</td>
    </tr>
</table>
