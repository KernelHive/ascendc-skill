# Heaviside
## 贡献说明

| 贡献者 | 贡献方 | 贡献算子        | 贡献时间      | 贡献内容          |
|-----|-----|-------------|-----------|---------------|
| 摆渡人 | 算子赛 | Heaviside算子 | 2025/6/12 | 新增Heaviside算子 |

## 支持的产品型号
- Atlas 200/500 A2 推理产品
- Atlas A2训练系列产品/Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  计算输入张量 input 的每个元素的 Heaviside 阶跃函数值。

- 原型信息

  <table>
    <tr>
        <th align="center">算子类型(OpType)</th><th colspan="5" align="center">Heaviside</th>
    </tr>
    <tr>
        <td rowspan="1" align="center"></td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">默认值</td>
    </tr>
        <tr><td rowspan="1" align="center">算子输入</td><td align="center">input</td><td align="center">-</td><td align="center">float32, float16</td><td align="center">ND</td><td align="center">\</td>
        <tr><td rowspan="1" align="center">算子输入</td><td align="center">values</td><td align="center">-</td><td align="center">float32, float16</td><td align="center">ND</td><td align="center">\</td>
    </tr>
        <tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center">-</td><td align="center">float32, float16</td><td align="center">ND</td><td align="center">\</td>
    </tr>
    <!-- <tr>
        <td rowspan="4" align="center">attr属性</td><td align="center">num_rows</td><td align="center">\</td><td align="center">int</td><td align="center">\</td><td align="center">\</td>
    </tr>
    <tr>
        <td align="center">num_columns</td><td align="center">\</td><td align="center">int</td><td align="center">\</td><td align="center">0</td>
    </tr>
    <tr>
        <td align="center">batch_shape</td><td align="center">\</td><td align="center">list_int</td><td align="center">\</td><td align="center">{1}</td>
    </tr>
    <tr>
        <td align="center">dtype</td><td align="center">\</td><td align="center">int</td><td align="center">\</td><td align="center">0</td>
    </tr> -->
    <tr>
        <td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">heaviside</td></td>
    </tr>
  </table>

## 约束与限制
- input，values，out的数据类型仅支持float32, float16，数据格式仅支持ND

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Heaviside算子。</td>
    </tr>


</table>
