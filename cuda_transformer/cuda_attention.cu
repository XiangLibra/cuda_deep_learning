#include <torch/extension.h>

__global__ void scaled_dot_product_attention(
    const float *Q, const float *K, const float *V, float *output, int batch_size, int seq_len, int d_model
) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < seq_len) {
        float score = 0.0;
        for (int i = 0; i < d_model; i++) {
            score += Q[batch * seq_len * d_model + row * d_model + i] * 
                     K[batch * seq_len * d_model + col * d_model + i];
        }
        score /= sqrtf(d_model);
        output[batch * seq_len * seq_len + row * seq_len + col] = score;
    }
}

// Python 調用的封裝函數
torch::Tensor attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int batch_size = Q.size(0);
    const int seq_len = Q.size(1);
    const int d_model = Q.size(2);

    auto output = torch::zeros({batch_size, seq_len, seq_len}, Q.device());

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   batch_size);

    scaled_dot_product_attention<<<numBlocks, threadsPerBlock>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, seq_len, d_model
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_cuda", &attention_cuda, "CUDA Attention Kernel");
}
