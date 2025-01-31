#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void multi_head_attention(
    const float *Q, const float *K, const float *V, float *output,
    int batch_size, int seq_len, int d_model, int nhead
) {
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = threadIdx.y;

    int head_size = d_model / nhead;

    __shared__ float shared_Q[512];  
    __shared__ float shared_K[512];

    if (row < seq_len && col < seq_len) {
        float score = 0.0;
        for (int i = 0; i < head_size; i++) {
            shared_Q[i] = Q[batch * seq_len * d_model + head * head_size + row * head_size + i];
            shared_K[i] = K[batch * seq_len * d_model + head * head_size + col * head_size + i];
            __syncthreads();
            score += shared_Q[i] * shared_K[i];
        }
        score /= sqrtf(head_size);
        output[batch * nhead * seq_len * seq_len + head * seq_len * seq_len + row * seq_len + col] = score;
    }
}

// 🚀 新增訓練函數 (更新權重)
__global__ void update_weights(float *weights, const float *grads, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grads[idx];  // SGD 更新權重
    }
}

// 🚀 CUDA 調用函數，新增訓練過程
extern "C" torch::Tensor attention_cuda_train(torch::Tensor Q, torch::Tensor K, torch::Tensor V, 
                                              torch::Tensor target, torch::Tensor weights, 
                                              int layers, int nhead, float lr) {
    const int batch_size = Q.size(0);
    const int seq_len = Q.size(1);
    const int d_model = Q.size(2);
    int head_size = d_model / nhead;

    auto output = torch::zeros({batch_size, nhead, seq_len, seq_len}, Q.device());

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x, nhead, batch_size);

    for (int epoch = 0; epoch < 10; epoch++) {  // 5 個 Epoch 訓練
        for (int i = 0; i < layers; i++) {  
            multi_head_attention<<<numBlocks, threadsPerBlock>>>(
                Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), output.data_ptr<float>(),
                batch_size, seq_len, d_model, nhead
            );
        }

        // 🚀 計算損失 (MSE)
        auto loss = (output - target).pow(2).mean();

        // 🚀 計算梯度 (簡化版)
        auto grads = 2 * (output - target) / target.numel();

        // 🚀 更新權重
        int weight_size = weights.numel();
        dim3 blockSize(256);
        dim3 gridSize((weight_size + blockSize.x - 1) / blockSize.x);
        update_weights<<<gridSize, blockSize>>>(weights.data_ptr<float>(), grads.data_ptr<float>(), lr, weight_size);

        // 🚀 顯示 Loss
        if (epoch % 1 == 0) {
            printf("Epoch %d: Loss = %f\n", epoch, loss.item<float>());
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_cuda_train", &attention_cuda_train, "CUDA Multi-Head Attention with Training");
}
