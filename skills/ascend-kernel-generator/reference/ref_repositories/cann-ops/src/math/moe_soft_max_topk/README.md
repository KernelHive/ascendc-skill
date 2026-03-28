# MoeSoftMaxTopk

## 贡献说明
| 贡献者   | 贡献方  | 贡献算子           | 贡献时间     | 贡献内容               |
|-------|------|----------------|----------|--------------------|
| yuuki | 社区任务 | MoeSoftMaxTopk | 2025/5/9 | 新增MoeSoftMaxTopk算子 |

## 支持的产品型号
- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  MoeSoftMaxTopk是softmax和topk的融合算子，其中softmax可以理解为对x计算最后一维每个数据的概率，在计算结果中筛选出k个最大结果，输出对应的y值和索引indices。  
 
- 原型信息

  <table>
    <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">MoeSoftMaxTopk</td></tr>
    </tr>
    <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>
    <tr><td align="center">x</td><td align="center">1024 * 16</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
    
    </tr>
    </tr>
    <tr><td rowspan="3" align="center">算子输出</td>
    
    <tr><td align="center">y</td><td align="center">1024 * 4</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
    <tr><td align="center">indices</td><td align="center">1024 * 4</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">attr属性</td><td align="center">k</td><td align="center">\</td><td align="center">int</td><td align="center">\</td><td align="center">4</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">moe_soft_max_topk</td></tr>
  </table>

## 约束与限制
- x,y,indices的数据类型仅支持float，数据格式仅支持ND

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用MoeSoftMaxTopk算子。</td>
    </tr>
</table>
