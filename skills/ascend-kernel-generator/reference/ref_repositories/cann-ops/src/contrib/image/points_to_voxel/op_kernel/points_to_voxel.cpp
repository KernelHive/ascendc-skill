/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file points_to_voxel.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class KernelPointsToVoxel
{
public:
    __aicore__ inline KernelPointsToVoxel() {}
    __aicore__ inline void Init(
        GM_ADDR points, 
        GM_ADDR voxels_out, 
        GM_ADDR coors_out, 
        GM_ADDR num_points_per_voxel, 
        GM_ADDR voxel_num, 
        GM_ADDR workspace,
        uint32_t ALIGN_NUM,
        uint32_t block_size,
        uint32_t core_size,
        uint32_t core_remain,
        int32_t ndimlength,
        int32_t Nlength,
        float voxel_size_x,
        float voxel_size_y,
        float voxel_size_z,
        float coors_range_xL,
        float coors_range_yL,
        float coors_range_zL,
        int32_t voxelmap_shape_xR,
        int32_t voxelmap_shape_yR,
        int32_t voxelmap_shape_zR,
        int32_t max_points,
        bool reverse_index,
        int32_t max_voxels)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->ndimlength = ndimlength;
        this->Nlength = Nlength;
        if(ALIGN_NUM != 0) {
            this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
            this->NlengthAlign = ((Nlength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
        }    
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        this->startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;
        this->voxelmap_length = voxelmap_shape_xR * voxelmap_shape_yR * voxelmap_shape_zR;
        this->voxelmap_shape_xR = voxelmap_shape_xR;
        this->voxelmap_shape_yR = voxelmap_shape_yR;
        this->voxelmap_shape_zR = voxelmap_shape_zR;
        this->max_points = max_points;
        this->reverse_index = reverse_index;
        this->max_voxels = max_voxels;

        tmp_global.SetGlobalBuffer((__gm__ int32_t *)workspace + this->startPointer, this->NlengthAlign * 4);
        points_global.SetGlobalBuffer((__gm__ float *)points + this->startPointer, ndimlength * Nlength + ALIGN_NUM);
        voxels_out_global.SetGlobalBuffer((__gm__ float *)voxels_out, max_voxels * max_points * ndimlength);
        coors_out_global.SetGlobalBuffer((__gm__ int32_t *)coors_out, max_voxels * 3);
        num_points_per_voxel_global.SetGlobalBuffer((__gm__ int32_t *)num_points_per_voxel, max_voxels);
        voxel_num_global.SetGlobalBuffer((__gm__ int32_t *)voxel_num, 1);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueX, BUFFER_NUM, this->tileLength * sizeof(int32_t));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(int32_t));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(int32_t));
        pipe.InitBuffer(outQueueRes, BUFFER_NUM, this->tileLength * sizeof(int32_t));

        pipe.InitBuffer(coor_range_Buf, (3 * this->tileLength) * sizeof(float));
        pipe.InitBuffer(voxel_size_Buf, (3 * this->tileLength) * sizeof(float));
        pipe.InitBuffer(QueueTmp1, this->tileLength * sizeof(int32_t));
        pipe.InitBuffer(QueueTmp2, this->tileLength * sizeof(int32_t));
        pipe.InitBuffer(QueueTmp3, this->tileLength * sizeof(int32_t));
        pipe.InitBuffer(offset_buf, this->tileLength / 4 * sizeof(uint32_t));

        LocalTensor<float> coor_range_Local = coor_range_Buf.Get<float>();
        LocalTensor<float> voxel_size_Local = voxel_size_Buf.Get<float>();
        coors_range_x_local = coor_range_Local[0];
        coors_range_y_local = coor_range_Local[this->tileLength];
        coors_range_z_local = coor_range_Local[this->tileLength * 2];
        voxel_size_x_local = voxel_size_Local[0];
        voxel_size_y_local = voxel_size_Local[this->tileLength];
        voxel_size_z_local = voxel_size_Local[this->tileLength * 2];
        Duplicate(coors_range_x_local, coors_range_xL, this->tileLength);
        Duplicate(coors_range_y_local, coors_range_yL, this->tileLength);
        Duplicate(coors_range_z_local, coors_range_zL, this->tileLength);
        Duplicate(voxel_size_x_local, voxel_size_x, this->tileLength);
        Duplicate(voxel_size_y_local, voxel_size_y, this->tileLength);
        Duplicate(voxel_size_z_local, voxel_size_z, this->tileLength);

        srcOffsetLocal = offset_buf.Get<uint32_t>();
        for(int32_t k = 0; k < (this->tileLength / 4); k++) {
            srcOffsetLocal.SetValue(k, k * 32);
        }
    }

    __aicore__ inline void Process(GM_ADDR workspace)
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount-1, length);
        Compute(loopCount-1, length);
        CopyOut(loopCount-1, length);

        SyncAll<true>();
        if (GetBlockIdx() == 0) {
            tmp_global.SetGlobalBuffer((__gm__ int32_t *)workspace, this->NlengthAlign * 4 + this->voxelmap_length + 512);
            voxelmap_global = tmp_global[this->NlengthAlign * 4 + 256];
            InitVoxelmap();
            InitNumpoints();
            GetResults();
        } else {
            return;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t process, uint32_t length)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, points_global[process * this->tileLength], length);
        inQueueX.EnQue(xLocal);
	    PipeBarrier<PIPE_ALL>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        DataCopy(yLocal, points_global[(process * this->tileLength) + this->Nlength], length);
        inQueueY.EnQue(yLocal);
	    PipeBarrier<PIPE_ALL>();
        LocalTensor<float> zLocal = inQueueZ.AllocTensor<float>();
        DataCopy(zLocal, points_global[(process * this->tileLength) + (this->Nlength * 2)], length);        
        inQueueZ.EnQue(zLocal);
	    PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Compute(uint32_t process, uint32_t length)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = inQueueZ.DeQue<float>();
        LocalTensor<int32_t> coorX = outQueueX.AllocTensor<int32_t>();
        LocalTensor<int32_t> coorY = outQueueY.AllocTensor<int32_t>();
        LocalTensor<int32_t> coorZ = outQueueZ.AllocTensor<int32_t>();
        LocalTensor<int32_t> resX = outQueueRes.AllocTensor<int32_t>();
        LocalTensor<int32_t> resTemp = QueueTmp1.Get<int32_t>();
		
	    PipeBarrier<PIPE_ALL>();
        Sub(xLocal, xLocal, coors_range_x_local, length);
        Div(xLocal, xLocal, voxel_size_x_local, length);
        Cast(coorX, xLocal, RoundMode::CAST_FLOOR, length);
        CompareGrid(coorX, this->voxelmap_shape_xR, length, resX);

	    PipeBarrier<PIPE_ALL>();
        Sub(yLocal, yLocal, coors_range_y_local, length);
        Div(yLocal, yLocal, voxel_size_y_local, length);
        Cast(coorY, yLocal, RoundMode::CAST_FLOOR, length);
        CompareGrid(coorY, this->voxelmap_shape_yR, length, resTemp);
        Mul(resX, resX, resTemp, length);

	    PipeBarrier<PIPE_ALL>();
        Sub(zLocal, zLocal, coors_range_z_local, length);
        Div(zLocal, zLocal, voxel_size_z_local, length);
        Cast(coorZ, zLocal, RoundMode::CAST_FLOOR, length);
        CompareGrid(coorZ, this->voxelmap_shape_zR, length, resTemp);
        Mul(resX, resX, resTemp, length);

	    PipeBarrier<PIPE_ALL>();
        outQueueX.EnQue(coorX);
        outQueueY.EnQue(coorY);
        outQueueZ.EnQue(coorZ);
        outQueueRes.EnQue(resX);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueZ.FreeTensor(zLocal);
    }

    __aicore__ inline void CopyOut(uint32_t process, uint32_t length)
    {
        LocalTensor<int32_t> coorX = outQueueX.DeQue<int32_t>();
        LocalTensor<int32_t> coorY = outQueueY.DeQue<int32_t>();
        LocalTensor<int32_t> coorZ = outQueueZ.DeQue<int32_t>();
        LocalTensor<int32_t> resX = outQueueRes.DeQue<int32_t>();
        DataCopy(tmp_global[process * this->tileLength], resX, length);
	    PipeBarrier<PIPE_ALL>();
        DataCopy(tmp_global[process * this->tileLength + this->NlengthAlign], coorX, length);
        DataCopy(tmp_global[process * this->tileLength + (this->NlengthAlign * 2)], coorY, length);
        DataCopy(tmp_global[process * this->tileLength + (this->NlengthAlign * 3)], coorZ, length);
	    PipeBarrier<PIPE_ALL>();
        outQueueX.FreeTensor(coorX);
        outQueueY.FreeTensor(coorY);
        outQueueZ.FreeTensor(coorZ);
        outQueueRes.FreeTensor(resX);
    }
    __aicore__ inline void CompareGrid(LocalTensor<int32_t> &src, int32_t upper_bound, uint32_t length, LocalTensor<int32_t> &res) 
    {
        LocalTensor<int32_t> tmp2 = QueueTmp2.Get<int32_t>();
        LocalTensor<int32_t> restmp = QueueTmp3.Get<int32_t>();
        Duplicate(tmp2, int32_t(upper_bound), length);
        Min(res, src, tmp2, length);
        Sub(tmp2, tmp2, res, length);
        Mins(res, tmp2, int32_t(1), length);

        Duplicate(tmp2, int32_t(0), length);
        Min(restmp, src, tmp2, length);
        Sub(tmp2, tmp2, restmp, length);
        Mins(restmp, tmp2, int32_t(1), length);
        Duplicate(tmp2, int32_t(1), length);
        Sub(restmp, tmp2, restmp, length);

        Mul(res, res, restmp, length);
	    PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void InitVoxelmap() 
    {
        LocalTensor<int32_t> coorX = outQueueX.AllocTensor<int32_t>();
        Duplicate(coorX, int32_t(-1), this->tileLength);
        int32_t loopCount = this->voxelmap_length / this->tileLength;
        for(int32_t k = 0; k < loopCount; k++) {
            DataCopy(voxelmap_global[k * this->tileLength], coorX, this->tileLength);
        }
        int32_t lengthremain = this->voxelmap_length % this->tileLength;
        if(lengthremain > 0) {
            uint32_t blockLen = lengthremain * sizeof(int32_t);
            DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
            DataCopyPad(voxelmap_global[loopCount * this->tileLength], coorX, copyParams);
        }
        outQueueX.FreeTensor(coorX);
    }
    __aicore__ inline void InitNumpoints()
    {
        LocalTensor<int32_t> coorY = outQueueY.AllocTensor<int32_t>();
        Duplicate(coorY, int32_t(0), this->tileLength);
        int32_t loopCount = this->max_voxels / this->tileLength;
        for(int32_t k = 0; k < loopCount; k++) {
            DataCopy(num_points_per_voxel_global[k * this->tileLength], coorY, this->tileLength);
        }
        int32_t lengthremain = this->max_voxels % this->tileLength;
        if(lengthremain > 0) {
            uint32_t blockLen = lengthremain * sizeof(int32_t);
            DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
            DataCopyPad(num_points_per_voxel_global[loopCount * this->tileLength], coorY, copyParams);
        }
        outQueueY.FreeTensor(coorY);
    }
    __aicore__ inline void GetResults()
    {
        int32_t voxel_num = 0;
        for(int32_t k = 0; k < this->Nlength; k++) {
            int32_t coor_to_voxelflag = tmp_global.GetValue(k);
            if(coor_to_voxelflag == 1) {  
                int32_t cooridx = tmp_global.GetValue(k + this->NlengthAlign) * this->voxelmap_shape_yR * this->voxelmap_shape_zR +
                                    tmp_global.GetValue(k + (this->NlengthAlign * 2)) * this->voxelmap_shape_zR + 
                                    tmp_global.GetValue(k + (this->NlengthAlign * 3));                
                int32_t voxelidx = voxelmap_global.GetValue(cooridx);
                if(voxelidx == -1) {
                    voxelidx = voxel_num;
                    if (voxel_num < this->max_voxels) {
                        voxel_num += 1;
                        voxelmap_global.SetValue(cooridx, voxelidx);
                        CopyOutCoors(voxelidx, k);
                    }
                }
                int32_t num = num_points_per_voxel_global.GetValue(voxelidx);
                if (num < this->max_points) {
                    CopyOutVoxelsGS(k, voxelidx, num);
                    num_points_per_voxel_global.SetValue(voxelidx, num + 1);
                }
            }
        }
        voxel_num_global.SetValue(0, voxel_num);
    }
    __aicore__ inline void CopyOutVoxelsGS(int32_t pos, int32_t voxelidx, int32_t num) 
    {
        for(int32_t k = 0; k < this->ndimlength; k++) {
            float tmp = points_global.GetValue(pos + (k * this->Nlength));
            voxels_out_global.SetValue(((voxelidx * this->max_points * this->ndimlength) + (num * this->ndimlength) + k), tmp);
        }
    }
    __aicore__ inline void CopyOutCoors(int32_t voxelidx, int32_t npos)
    {
        if(this->reverse_index == false) {
            coors_out_global.SetValue(voxelidx * 3, tmp_global.GetValue(npos + this->NlengthAlign));
            coors_out_global.SetValue(voxelidx * 3 + 1, tmp_global.GetValue(npos + (this->NlengthAlign * 2)));
            coors_out_global.SetValue(voxelidx * 3 + 2, tmp_global.GetValue(npos + (this->NlengthAlign * 3)));
        } else {
            coors_out_global.SetValue(voxelidx * 3, tmp_global.GetValue(npos + (this->NlengthAlign * 3)));
            coors_out_global.SetValue(voxelidx * 3 + 1, tmp_global.GetValue(npos + (this->NlengthAlign * 2)));
            coors_out_global.SetValue(voxelidx * 3 + 2, tmp_global.GetValue(npos + this->NlengthAlign));
        }         
    }

private:
    GlobalTensor<float> points_global;
    GlobalTensor<float> voxels_out_global;
    GlobalTensor<int32_t> coors_out_global; 
    GlobalTensor<int32_t> num_points_per_voxel_global;
    GlobalTensor<int32_t> voxel_num_global;
    GlobalTensor<int32_t> tmp_global;
    GlobalTensor<int32_t> voxelmap_global;
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY, inQueueZ;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueX, outQueueY, outQueueZ, outQueueRes;
    TBuf<QuePosition::VECCALC> coor_range_Buf, voxel_size_Buf;
    TBuf<QuePosition::VECCALC> QueueTmp1, QueueTmp2, QueueTmp3;
    TBuf<QuePosition::VECCALC> offset_buf;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    int32_t ndimlength;
    int32_t Nlength;
    int32_t voxelmap_shape_xR;
    int32_t voxelmap_shape_yR;
    int32_t voxelmap_shape_zR;
    int32_t max_points;
    bool reverse_index;
    int32_t max_voxels;
    int32_t voxelmap_length;
    uint32_t startPointer;
    int32_t NlengthAlign;

    LocalTensor<float> coors_range_x_local;
    LocalTensor<float> coors_range_y_local;
    LocalTensor<float> coors_range_z_local;
    LocalTensor<float> voxel_size_x_local;
    LocalTensor<float> voxel_size_y_local;
    LocalTensor<float> voxel_size_z_local;
    LocalTensor<uint32_t> srcOffsetLocal;
};
extern "C" __global__ __aicore__ void points_to_voxel(GM_ADDR points, GM_ADDR voxels_out, GM_ADDR coors_out, GM_ADDR num_points_per_voxel, GM_ADDR voxel_num, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelPointsToVoxel<DTYPE_POINTS> op;
    op.Init(points, voxels_out, coors_out, num_points_per_voxel, voxel_num, workspace,
            tiling_data.ALIGN_NUM,
            tiling_data.block_size,
            tiling_data.core_size,
            tiling_data.core_remain,
            tiling_data.ndimlength,
            tiling_data.Nlength,
            tiling_data.voxel_size_x,
            tiling_data.voxel_size_y,
            tiling_data.voxel_size_z,
            tiling_data.coors_range_xL,
            tiling_data.coors_range_yL,
            tiling_data.coors_range_zL,
            tiling_data.voxelmap_shape_xR,
            tiling_data.voxelmap_shape_yR,
            tiling_data.voxelmap_shape_zR,
            tiling_data.max_points,
            tiling_data.reverse_index,
            tiling_data.max_voxels);
    op.Process(workspace);
}

