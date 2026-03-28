# PreLayerNorm

## 贡献说明
| 贡献者   | 贡献方  | 贡献算子         | 贡献时间      | 贡献内容             |
|-------|------|--------------|-----------|------------------|
| yuuki | 社区任务 | PreLayerNorm | 2025/4/30 | 新增PreLayerNorm算子 |

## 支持的产品型号

- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述

- 功能描述

  PreLayerNorm是Add和LayerNorm的融合算子，Add算子的输出作为LayerNorm算子的第一个输入。对输入x, y先相加得到的数据，根据系数beta 和偏置gamma使其Add(x, y)的值收敛到固定区间。  

- 原型信息

  <table>
    <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">PreLayerNorm</td></tr>
    </tr>
    <tr><td rowspan="5" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">默认值</td></tr>
    <tr><td align="center">x</td><td align="center">4980 * 4 * 2048</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
    <tr><td align="center">y</td><td align="center">4980 * 4 * 2048</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
    <tr><td align="center">gamma</td><td align="center">2048</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
    <tr><td align="center">beta</td><td align="center">2048</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">4980 * 4 * 2048</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">attr属性</td><td align="center">epsilon</td><td align="center">\</td><td align="center">double</td><td align="center">\</td><td align="center">1e-5</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">pre_layer_norm</td></tr>
  </table>

## 约束与限制
- x,y,gamma,beta,z,out的数据类型仅支持float，数据格式仅支持ND

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用PreLayerNorm算子。</td>
    </tr>
</table>
