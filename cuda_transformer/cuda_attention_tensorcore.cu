#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;  // NVIDIA Tensor Core API

// 🚀 核心函數：使用 Tensor Cores 計算 Attention
__global__ void tensor_core_attention(
    const half *Q, const half *K, const half *V, half *output,
    int batch_size, int seq_len, int d_model, int nhead
) {
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = threadIdx.y;

    int head_size = d_model / nhead;

    // ✅ 確保記憶體對齊 (使用 alignas(16) 來對齊)
    alignas(16) __shared__ half shared_Q[512 * 16];
    alignas(16) __shared__ half shared_K[512 * 16];

    if (row < seq_len && col < seq_len) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_Q;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_K;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_output;

        // 🚀 使用 Tensor Cores `wmma::load_matrix_sync()` 載入 FP16 矩陣
        wmma::load_matrix_sync(frag_Q, Q + batch * seq_len * d_model + head * head_size + row * head_size, head_size);
        wmma::load_matrix_sync(frag_K, K + batch * seq_len * d_model + head * head_size + col * head_size, head_size);
        
        wmma::fill_fragment(frag_output, __float2half(0.0f));

        wmma::mma_sync(frag_output, frag_Q, frag_K, frag_output);
        
        wmma::store_matrix_sync(output + batch * nhead * seq_len * seq_len + head * seq_len * seq_len + row * seq_len + col, frag_output, seq_len, wmma::mem_row_major);
    }
}

// 🚀 Python 調用函數
extern "C" torch::Tensor attention_tensor_core(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int layers, int nhead) {
    const int batch_size = Q.size(0);
    const int seq_len = Q.size(1);
    const int d_model = Q.size(2);
    int head_size = d_model / nhead;

    // ✅ 確保 Tensor 記憶體對齊，避免 `misaligned address`
    Q = Q.to(torch::kHalf).contiguous();
    K = K.to(torch::kHalf).contiguous();
    V = V.to(torch::kHalf).contiguous();

    auto output = torch::zeros({batch_size, nhead, seq_len, seq_len}, torch::TensorOptions().dtype(torch::kHalf).device(Q.device()));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x, nhead, batch_size);

    for (int i = 0; i < layers; i++) {  
        tensor_core_attention<<<numBlocks, threadsPerBlock>>>(
            reinterpret_cast<const half*>(Q.data_ptr<c10::Half>()),
            reinterpret_cast<const half*>(K.data_ptr<c10::Half>()),
            reinterpret_cast<const half*>(V.data_ptr<c10::Half>()),
            reinterpret_cast<half*>(output.data_ptr<c10::Half>()),
            batch_size, seq_len, d_model, nhead
        );
    }

    return output.to(torch::kFloat);  // 🚀 轉回 FP32 以供 PyTorch 使用
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_tensor_core", &attention_tensor_core, "CUDA Tensor Core Multi-Head Attention (24 層, 8 Heads)");
}
