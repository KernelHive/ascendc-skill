# RoiAlignRotated

## 支持的产品型号

Atlas 训练系列产品/Atlas 推理系列产品/Atlas A2训练系列产品/Atlas 800I A2推理产品/Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：RoiAlignRotated是对于给定的Roi（Region of interest）在特征图上提取对齐地特征，其首先将Roi区域划分为若干个更小的网格，然后在每个网格内执行双线性插值，得到固定大小的特征。

  **说明：**
  无。

## 实现原理

- 1.该算子kernel对Roi个数进行分核，每个核内再按照256进行循环，一次处理256个roi。
- 2.然后就是针对循环内的每个Roi进行:
-   ①.scale
-   ②.计算在特征图中的起始位置
-   ③.计算bin_size
-   ④.计算双线性插值
- 3.搬出该roi的特征

## 约束与限制

- 输入format为NCHW，其中N仅支持1维，C仅支持3维，属性值aligned与clockwise仅支持True。

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">RoiAlignRotated</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format和说明</td></tr>
<tr><td align="center">x</td><td align="center">(N,C,H,W)</td><td align="center">float32</td><td align="center">NC1HWC0/特征图</td></tr>
<tr><td align="center">rois</td><td align="center">(N,6)</td><td align="center">int32</td><td align="center">ND/roi框，6维分别为(batch_idx,center_x,center_y,w,h,angle)</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">(N,C,H,W)</td><td align="center">int32</td><td align="center">NC1HWC0/roialign后的特征图</td></tr>
</tr>
<tr><td rowspan="6" align="center">属性输入</td><td align="center">pooled_h</td><td align="center"></td><td align="center">int</td><td align="center">输出特征图的高度</td></tr>
<tr><td align="center">pooled_w</td><td align="center"></td><td align="center">int</td><td align="center">输出特征图的宽度</td></tr>
<tr><td align="center">spatial_scale</td><td align="center"></td><td align="center">float</td><td align="center">对输入框进行缩放</td></tr>
<tr><td align="center">sampling_ratio</td><td align="center"></td><td align="center">int</td><td align="center">每个输出对于输入的采样个数，默认为0</td></tr>
<tr><td align="center">aligned</td><td align="center"></td><td align="center">bool</td><td align="center">是否更好的输出对齐结果，默认True</td></tr>
<tr><td align="center">clockwise</td><td align="center"></td><td align="center">bool</td><td align="center">时针方向，默认False</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">roi_align_rotated</td></tr>
</table>
