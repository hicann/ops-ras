/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_abft_verify.cpp
 * \brief
 */

#include "matmul_abft_verify.h"
#include "matmul_abft_verify_tiling_key.h"
#include "lib/matmul_intf.h"

#include "FT_self/gemm/block/block_mmad.hpp"
#include "FT_self/gemv/block/block_gemv.hpp"

#include "FT_self/gemv/tile/tile_fault_sum.hpp"
#include "FT_self/gemv/block/block_gemv.hpp"
#include "FT_self/gemv/tile/tile_matmul_elem_add.hpp"
#include "FT_self/gemv/tile/tile_threshold.hpp"      // TileThreCalc
#include "FT_self/gemv/tile/tile_std_estimate.hpp"   // TileStdEstRobust
#include "FT_self/gemv/helper.hpp"

#include "FT_self/gemm/tile/gemm_tile_copy.hpp"
#include "FT_self/gemm/tile/tile_mmad.hpp"
#include "FT_self/gemv/tile/gemv_tile_copy.hpp"

#include "FT_self/gemv/tile/tile_vmuls.hpp"
#include "FT_self/gemv/helper.hpp"

using GemmInTypeC = DTYPE_A;
using GemmInTypeN = DTYPE_A;

using GemmOutTypeC = float;
using GemmOutTypeN = float;

#if (ORIG_DTYPE_A == DT_FLOAT16)
using GemvInTypeCforCE = DTYPE_A;
using GemvInTypeNforCE = DTYPE_A;
#else
using GemvInTypeCforCE = float;
using GemvInTypeNforCE = float;
#endif

using GemvInTypeCforAB = DTYPE_CHECKSUM_WEIGHT;
using GemvInTypeNforAB = DTYPE_CHECKSUM_WEIGHT;


using GemvOutTypeC = float;
using GemvOutTypeN = float;

using ScalarTypeC = float;
using ScalarTypeN = float;

using ScalarType = float;

constexpr bool enableUnitFlag = false;
constexpr bool enableShuffleK = true;
constexpr uint64_t WORKSPACE_ALIGNMENT = 32;

__aicore__ inline uint64_t AlignWorkspaceOffset(uint64_t offset)
{
    return (offset + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
}

__aicore__ inline GM_ADDR TakeWorkspaceTensor(
    GM_ADDR workspace, uint64_t &offset, uint64_t elements, uint64_t elementBytes)
{
    offset = AlignWorkspaceOffset(offset);
    GM_ADDR tensor = workspace + offset;
    offset += elements * elementBytes;
    return tensor;
}

template <uint32_t schMode>
__global__ __aicore__ void matmul_abft_verify(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR checksumWeight,
    GM_ADDR compRow, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MatmulAbftVerifyTilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulAbftVerifyTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);

    const uint64_t m = tilingData.compute.problemGemmShape.m;
    const uint64_t k = tilingData.compute.problemGemmShape.k;
    const uint64_t splitN = tilingData.compute.splitNNum;
    const uint64_t rowSplitElements = m * splitN;
    const uint64_t bStatElements = ((splitN + 7) / 8) * 8 + 8;
    const uint64_t beElements = k * splitN;
#if (ORIG_DTYPE_A == DT_FLOAT)
    constexpr uint64_t precisionBytes = sizeof(float);
#else
    constexpr uint64_t precisionBytes = sizeof(uint16_t);
#endif
#if (ORIG_DTYPE_A == DT_FLOAT16)
    constexpr uint64_t ceBytes = sizeof(uint16_t);
#else
    constexpr uint64_t ceBytes = sizeof(float);
#endif

    // Workspace ABI. Keep this order synchronized with host tiling and the operator document.
    uint64_t workspaceOffset = 0;
    GM_ADDR zRow = TakeWorkspaceTensor(userWorkspace, workspaceOffset, rowSplitElements, sizeof(float));
    GM_ADDR dRow = TakeWorkspaceTensor(userWorkspace, workspaceOffset, rowSplitElements, sizeof(float));
    GM_ADDR threshold = TakeWorkspaceTensor(userWorkspace, workspaceOffset, rowSplitElements, sizeof(float));
    GM_ADDR bMeanAbs = TakeWorkspaceTensor(userWorkspace, workspaceOffset, bStatElements, sizeof(float));
    GM_ADDR bMeanSquare = TakeWorkspaceTensor(userWorkspace, workspaceOffset, bStatElements, sizeof(float));
    GM_ADDR bVar = TakeWorkspaceTensor(userWorkspace, workspaceOffset, bStatElements, sizeof(float));
    GM_ADDR be = TakeWorkspaceTensor(userWorkspace, workspaceOffset, beElements, precisionBytes);
    GM_ADDR beForAiv = TakeWorkspaceTensor(userWorkspace, workspaceOffset, beElements, ceBytes);
    GM_ADDR bMaxSlice = TakeWorkspaceTensor(userWorkspace, workspaceOffset, beElements, sizeof(float));
    GM_ADDR bMinSlice = TakeWorkspaceTensor(userWorkspace, workspaceOffset, beElements, sizeof(float));
    GM_ADDR aMax = TakeWorkspaceTensor(userWorkspace, workspaceOffset, m, sizeof(float));
    GM_ADDR aMean = TakeWorkspaceTensor(userWorkspace, workspaceOffset, m, sizeof(float));
    GM_ADDR aMin = TakeWorkspaceTensor(userWorkspace, workspaceOffset, m, sizeof(float));
    GM_ADDR internalWorkspace = userWorkspace + AlignWorkspaceOffset(workspaceOffset);

#if (ORIG_DTYPE_A == DT_FLOAT16)
    using TilingPolicy = MatmulAbftVerifyFp16TilingPolicy;
#elif (ORIG_DTYPE_A == DT_FLOAT)
    using TilingPolicy = MatmulAbftVerifyFp32TilingPolicy;
#else
    using TilingPolicy = MatmulAbftVerifyBf16TilingPolicy;
#endif

    union AMeanFactorStorage {
        uint32_t bits;
        GemmInTypeN value;
    } aMeanFactorStorage;
    aMeanFactorStorage.bits = tilingData.compute.aMeanFactorBits;
    const GemmInTypeN aMeanFactor = aMeanFactorStorage.value;
    using L1TileShape =
        FTSelf::GemmShape<128, 256, TilingPolicy::L1TileShape_K>;
    using L0TileShape =
        FTSelf::GemmShape<128, 256, TilingPolicy::L0TileShape_K>;

    using LayoutX = FTSelf::layout::VectorLayout;
    using LayoutY = FTSelf::layout::VectorLayout;
    using LayoutCOMP = FTSelf::layout::VectorLayout;

    using LayoutA = FTSelf::layout::RowMajor;
    using LayoutACol = FTSelf::layout::ColumnMajor;

    using LayoutB = FTSelf::layout::RowMajor;
    using LayoutBCol = FTSelf::layout::ColumnMajor;

    using LayoutC = FTSelf::layout::RowMajor;
    using LayoutCCol = FTSelf::layout::ColumnMajor;

    using LayoutZ = FTSelf::layout::VectorLayout;
    using FT_COMP_TYPE = FTSelf::Gemv::helper::FT_COMP_TYPE;


    using ArchTag = FTSelf::Arch::AtlasA2;
    using FT_REDUCE_TYPE = FTSelf::Gemv::helper::FT_REDUCE_TYPE;

    using LayoutMY = FTSelf::layout::RowMajor;
    using LayoutMX = FTSelf::layout::RowMajor;

    using FT_ENC_TYPE = FTSelf::Gemv::helper::FT_ENC_TYPE;
    using FT_RCE_THRE_TYPE = FTSelf::Gemv::helper::FT_RCE_THRE_TYPE;
    using FT_L02L1_TYPE = FTSelf::Gemv::helper::FT_L02L1_TYPE;
    using FT_AIC_BE_SCHEME = FTSelf::Gemv::helper::FT_AIC_BE_SCHEME;

    #if (ORIG_DTYPE_A == DT_FLOAT16 || ORIG_DTYPE_A == DT_FLOAT)
        constexpr FT_AIC_BE_SCHEME BlockFTGemvAIC_BE_SCHEME = FT_AIC_BE_SCHEME::ROWCOMPLETE;
        constexpr FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE BlockFTSum_A_B = FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED;
    #else
        constexpr FT_AIC_BE_SCHEME BlockFTGemvAIC_BE_SCHEME = FT_AIC_BE_SCHEME::ROWCOMPLETE_BF;
        constexpr FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE BlockFTSum_A_B = FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED_BF;
    #endif


    // using L1TileShapeC = GemvShape<32, 512>;
    // using L0TileShapeC = GemvShape<32, 256>;

    using L1TileShapeAB = FTSelf::GemmShape<128, 128, 128>;
    using L0TileShapeAB = FTSelf::GemmShape<128, 128, 128>;

    using L1TileShapeBE =
        FTSelf::GemvShape<TilingPolicy::L1TileShapeBE_M, 256>;
    using L0TileShapeBE =
        FTSelf::GemvShape<TilingPolicy::L0TileShapeBE_M, 64>;

    using UBBlockShapeBE = FTSelf::GemvShape<L1TileShapeBE::M*2, L1TileShapeBE::N*1>;

    using AType = FTSelf::GemmType<GemmInTypeN, LayoutA>;
    using BType = FTSelf::GemmType<GemmInTypeN, LayoutB>;

    using CType = FTSelf::GemmType<GemmOutTypeN, LayoutC>;

    using XTypeAIC = FTSelf::GemmType<GemvInTypeNforAB, LayoutX>;
    using YTypeBEAIC = FTSelf::GemmType<GemvInTypeNforAB, LayoutMY>;
    using BiasType = void;

    using XTypeAIV = FTSelf::GemmType<GemvInTypeNforCE, LayoutX>;

    using GemvDispatchPolicy = FTSelf::Gemv::GemvAtlasA2;
    using COMPDispatchPolicy = FTSelf::Gemv::GemvAtlasA2;
    using BeAICDispatchPolicy = FTSelf::Gemv::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using GEMVAICDispatchPolicy = FTSelf::Gemv::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using TileCopyGemvAic = FTSelf::Gemv::Tile::TileCopyGemvAic<typename BeAICDispatchPolicy::ArchTag, BType, XTypeAIC, YTypeBEAIC, BiasType>;
    using TileMmadGemvAic = FTSelf::Gemm::Tile::TileMmad<FTSelf::Arch::AtlasA2, XTypeAIC, BType, BiasType>;

    using BlockFTGemvAIC = FTSelf::Gemv::Block::BlockFTGemvBe<BeAICDispatchPolicy,
        BlockFTGemvAIC_BE_SCHEME,
        UBBlockShapeBE, L1TileShapeBE, L0TileShapeBE,
        BType, XTypeAIC, YTypeBEAIC, BiasType, TileCopyGemvAic, TileMmadGemvAic>;

    using ZType = FTSelf::GemmType<GemvOutTypeN, LayoutZ>;

    static constexpr FT_AIC_BE_SCHEME BE_SCHEME = BlockFTGemvAIC::BE_SCHEME;

    constexpr uint32_t computeLength = 8192;

    using YType = FTSelf::GemmType<GemvInTypeNforCE, LayoutY>;

    using BRedType = FTSelf::GemmType<GemvInTypeNforCE, LayoutB>;
    using TileVmuls = FTSelf::Gemv::Tile::TileVmuls<ArchTag, XTypeAIV>;

    using MmadDispatchPolicy = FTSelf::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;


    using L1TileShapeFirst = FTSelf::GemmShape<
        256, 256, TilingPolicy::L1TileShapeFirst_K>;
    using L0TileShapeFirst = FTSelf::GemmShape<
        256, 256, TilingPolicy::L0TileShapeFirst_K>;

    using L1TileShapeforFT = FTSelf::GemmShape<L1TileShapeFirst::M, 16, L1TileShapeFirst::K>;
    using L0TileShapeforFT = FTSelf::GemmShape<L0TileShapeFirst::M, 16, L0TileShapeFirst::K>;

    using MXType = FTSelf::GemmType<GemvInTypeNforAB, LayoutMX>;
    using MYType = FTSelf::GemmType<GemvOutTypeN, LayoutMY>;

    using BlockMmadABe = FTSelf::Gemm::Block::BlockMmadSpecABeNoSplitKRobust<
        MmadDispatchPolicy,
        L1TileShapeforFT,
        L0TileShapeforFT,
        AType, BType, CType, MXType, MYType, BiasType>;

    using BlockMmadPreload = FTSelf::Gemm::Block::BlockMmadPreload<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    using TileFaultCopyRedAiv = FTSelf::Gemv::Tile::TileCopyFTRedAiv<FTSelf::Gemv::Arch::AtlasA2,
        AType, BType, YType, ZType>;

    using UBTileShapeforB =
        FTSelf::GemvShape<TilingPolicy::UBTileShapeforB_M, L0TileShape::N>;
    using UBBlockShapeforB = FTSelf::GemvShape<UBTileShapeforB::M*2, UBTileShapeforB::N*2>;

    using UBTileShapeforA =
        FTSelf::GemvShape<TilingPolicy::UBTileShapeforA_M, 256>;

    using ARedType = FTSelf::GemmType<GemvInTypeNforCE, LayoutA>;
    using TileFaultSum = FTSelf::Gemv::Tile::TileFaultSum<FTSelf::Gemv::Arch::AtlasA2, FT_REDUCE_TYPE::MAX_MIN, ARedType, ZType>;

    using BlockFTSum = FTSelf::Gemv::Block::BlockFTSumNoSplitK<
        GemvDispatchPolicy,
        FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        BlockFTSum_A_B,
        UBTileShapeforB, UBBlockShapeforB, UBTileShapeforA,
        L1TileShape, AType, BType, YType, ZType, void,
        TileFaultCopyRedAiv, TileFaultSum>;

    using SliceSumDispatchPolicy = FTSelf::Gemv::GemvAtlasA2;

    using SliceSumUBTileShape = FTSelf::GemvShape<8, 256>;
    using TileMatrixAddforABEReduce = FTSelf::Gemv::Tile::TileMatmulAdd<
        typename SliceSumDispatchPolicy::ArchTag, MYType, MYType, void>;
    using TileCopyMatrixAddforABEReduce = FTSelf::Gemv::Tile::TileCopyMatrixAddAiv<
        typename SliceSumDispatchPolicy::ArchTag, MYType, MYType, void>;

    using BlockSliceSum = FTSelf::Gemv::Block::BlockSliceKMNSum<SliceSumDispatchPolicy,
        FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::ABE_FUSED_THRE,
        SliceSumUBTileShape, MYType, MYType, void, TileCopyMatrixAddforABEReduce, TileMatrixAddforABEReduce>;

    using MeanMaxTileVmuls = FTSelf::Gemv::Tile::TileVmuls<FTSelf::Arch::AtlasA2, ZType>;
    using UBTileShapeforBRed =
        FTSelf::GemvShape<TilingPolicy::UBTileShapeforBRed_M, TilingPolicy::UBTileShapeforBRed_N>;
    using TileFaultSumBReduce = FTSelf::Gemv::Tile::TileFaultSum<FTSelf::Gemv::Arch::AtlasA2, FT_REDUCE_TYPE::SUM_MAX, BRedType, ZType>;
    using TileFaultCopyBReduce = FTSelf::Gemv::Tile::TileCopyGemvAiv<FTSelf::Gemv::Arch::AtlasA2, BRedType, YType, ZType>;

    using BlockSliceRed = FTSelf::Gemv::Block::BlockSliceKMNSum<
        SliceSumDispatchPolicy,
        FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_ROBUST,
        SliceSumUBTileShape,
        UBTileShapeforBRed,
        MYType, BRedType, ZType,
        MYType, void,
        TileCopyMatrixAddforABEReduce,
        TileFaultCopyBReduce,
        TileMatrixAddforABEReduce,
        TileFaultSumBReduce,
        MeanMaxTileVmuls>;

    using UBTileShapeCE = FTSelf::GemvShape<64, L1TileShape::N>;
    using UBBlockShapeCE = FTSelf::GemvShape<L1TileShape::M, UBTileShapeCE::N>;

    using COMPZType = FTSelf::GemmType<uint8_t, LayoutZ>;
    using TileFaultCopyCE = FTSelf::Gemv::Tile::TileCopyGemvThreCompFusedAiv<FTSelf::Gemv::Arch::AtlasA2,
        CType, CType, XTypeAIV, ZType, COMPZType, void>;

    using ThreCalcDispatchPolicy = FTSelf::Gemv::GemvAtlasA2;
    using TileThreCalc = FTSelf::Gemv::Tile::TileThreCalc<
        typename ThreCalcDispatchPolicy::ArchTag,
        FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        CType, ZType, ZType, BiasType>;

    using TileStdEst = FTSelf::Gemv::Tile::TileStdEstRobust<
        typename ThreCalcDispatchPolicy::ArchTag,
        ZType,
        ZType
    >;

    using TileFaultSumCSum = FTSelf::Gemv::Tile::TileFaultSum<FTSelf::Gemv::Arch::AtlasA2, FT_REDUCE_TYPE::SUM, CType, ZType>;

    using BlockFTGemvAIV = FTSelf::Gemv::Block::BlockFTGemvCENoSplitKPreload<
        GemvDispatchPolicy,
        FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED,
        FTSelf::Gemv::helper::FT_ENC_TYPE::RCE,
        FTSelf::Gemv::helper::FT_COMP_TYPE::RSUB,
        FTSelf::Gemv::helper::FT_ABE_TYPE::CENTRAL_BLOCK,
        UBTileShapeCE, UBBlockShapeCE, L1TileShape,
        CType, XTypeAIV, ZType, COMPZType, void,
        TileFaultCopyCE, TileFaultSumCSum,
        TileThreCalc, TileStdEst>;

    using UBTileShape = FTSelf::GemmShape<L1TileShapeAB::M, L1TileShapeAB::N, L1TileShapeAB::K>;
    using TileFaultCopy = FTSelf::Gemv::Tile::TileCopyGemvAiv<FTSelf::Gemv::Arch::AtlasA2, AType, XTypeAIV, ZType>;
    using TileFaultVmad = FTSelf::Gemv::Tile::TileVmad<FTSelf::Gemv::Arch::AtlasA2, AType, XTypeAIV, ZType>;

    using BlockSumGemv = FTSelf::Gemv::Block::BlockSumGemv<GemvDispatchPolicy, UBTileShape, AType, XTypeAIV, ZType, void, TileFaultCopy, TileFaultVmad, TileVmuls>;

    using BlockConfig = MatfulFTKernel::BlockConfig<
        BlockMmadABe, BlockMmadPreload,
        BlockFTGemvAIC, BlockFTSum, BlockFTGemvAIV,
        BlockSliceRed, BlockSliceSum>;

    using MatmulFTKernel = MatfulFTKernel::MatmulAsVarABonAicSplitRobustPreload<BlockConfig>;

#if (ORIG_DTYPE_A == DT_FLOAT16) || (ORIG_DTYPE_A == DT_BF16) || (ORIG_DTYPE_A == DT_FLOAT)
    if constexpr (schMode == MATMUL_ABFT_VERIFY_TPL_SCH_MODE_BF16) {
        typename MatmulFTKernel::Arguments arguments{
            FTSelf::GemmCoord{
                tilingData.compute.problemGemmShape.m,
                tilingData.compute.problemGemmShape.n,
                tilingData.compute.problemGemmShape.k},
            FTSelf::GemvCoord{tilingData.compute.problemShape.m, tilingData.compute.problemShape.n},
            sizeof(GemmInTypeN),
            checksumWeight,
            a,
            b,
            c,
            zRow,
            dRow,
            compRow,
            be,
            beForAiv,
            bMaxSlice,
            bMinSlice,
            bMeanAbs,
            bMeanSquare,
            bVar,
            aMeanFactor,
            aMean,
            aMax,
            aMin,
            threshold,
            FT_ENC_TYPE::RCE,
            1,
            static_cast<bool>(tilingData.compute.outputWorkspace),
            0.0f,
            0.0f,
            tilingData.compute.eMax,
            static_cast<bool>(tilingData.compute.outputThre),
            static_cast<bool>(tilingData.compute.outputCE),
            tilingData.compute.splitKNum
        };

        typename MatmulFTKernel::Params params =
            MatmulFTKernel::ToUnderlyingArguments(arguments, internalWorkspace, tilingData.compute);
        MatmulFTKernel op;
        op(params);
    }
#endif
}
