/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef CSRC_ASCEND950_FLASH_ATTN_NPU_4_FA_METADATA_ARGS_H
#define CSRC_ASCEND950_FLASH_ATTN_NPU_4_FA_METADATA_ARGS_H

#include <cstdint>
#include <type_traits>
#include "tilingdata.h"

static_assert(std::is_trivially_copyable<FAInferTilingData>::value,
              "FAInferTilingData must be trivially copyable");
namespace fa_metadata {
constexpr uint32_t MASK_DIM = 2048;
constexpr uint64_t MASK_BYTES = uint64_t(MASK_DIM) * MASK_DIM;
inline uint64_t TilingOffset(bool mask) { return mask ? MASK_BYTES : 0; }
inline uint64_t MetadataBytes(bool mask) { return TilingOffset(mask) + sizeof(FAInferTilingData); }
constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = uint64_t(128) * 512;
constexpr uint32_t PRELAUNCH_NUM = 3;
inline uint64_t Mm1OutSize(uint64_t n) { return n * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELAUNCH_NUM; }
inline uint64_t SmOnlineOutSize(uint64_t n) { return n * WORKSPACE_BLOCK_SIZE_DB * 2 * PRELAUNCH_NUM; }
inline uint64_t Mm2OutSize(uint64_t n) { return n * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELAUNCH_NUM; }
inline uint64_t UpdateOutSize(uint64_t n) { return n * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELAUNCH_NUM; }
inline uint64_t WorkSpaceSize(uint64_t n) { return Mm1OutSize(n) + SmOnlineOutSize(n) + Mm2OutSize(n) + UpdateOutSize(n); }
}
struct FAMetadataArgs {
    uint64_t cuSeqlensQAddr, seqlensKAddr, metaOutAddr;
    uint32_t batch, numHeads, numHeadsK, embeddingSize, embeddingSizeV;
    uint32_t numBlocks, blockSize, maxNumBlocksPerBatch, maxQSeqlen;
    uint32_t maskType, blockDim, isVarlen, isVarlenKv, pagedKV, numSplits;
    float softmaxScale;
    int64_t windowSizeLeft, windowSizeRight;
};
#endif
