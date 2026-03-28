# BevPool
## 贡献说明
| 贡献者 | 贡献方  | 贡献算子    | 贡献时间      | 贡献内容        |
|-----|------|---------|-----------|-------------|
| 奇迹  | 社区任务 | BevPool | 2025/5/21 | 新增BevPool算子 |

## 支持的产品型号
- Atlas A2训练系列产品
- Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述
- 功能描述

  `BevPool`算子实现通过预计算视锥索引与体素索引的映射关系，避免显式存储视锥特征，从而将多摄像头图像特征高效聚合到BEV（鸟瞰图）空间。

- 原型信息

  <table>
    <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">BevPool</td></tr>
    </tr>
    <tr><td rowspan="8" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td align="center">depth</td><td align="center">-</td><td align="center">float16、float32</td><td align="center">ND</td></tr>
    <tr><td align="center">feat</td><td align="center">-</td><td align="center">float16、float32</td><td align="center">ND</td></tr>
    <tr><td align="center">ranks_depth</td><td align="center">-</td><td align="center">int32</td><td align="center">ND</td></tr>
    <tr><td align="center">ranks_feat</td><td align="center">-</td><td align="center">int32</td><td align="center">ND</td></tr>
    <tr><td align="center">ranks_bev</td><td align="center">-</td><td align="center">int32</td><td align="center">ND</td></tr>
    <tr><td align="center">interval_starts</td><td align="center">-</td><td align="center">int32</td><td align="center">ND</td></tr>
    <tr><td align="center">interval_lengths</td><td align="center">-</td><td align="center">int32</td><td align="center">ND</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center">-</td><td align="center">float16、float32</td><td align="center">ND</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">算子属性</td><td align="center">bev_feat_shape</td><td align="center">\</td><td align="center">list_int</td><td align="center">\</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">bev_pool</td></tr>
  </table>

## 约束与限制
- depth，feat，ranks_depth，ranks_feat，ranks_bev，interval_start，interval_lengths，out的数据类型仅支持float16、float32，数据格式仅支持ND

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用BevPool算子。</td>
    </tr>
</table>
