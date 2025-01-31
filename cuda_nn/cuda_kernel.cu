#include <torch/extension.h>

__global__ void matrix_mul_cuda(float *x, float *w, float *y, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += x[row * N + i] * w[i * K + col];
        }
        y[row * K + col] = sum;
    }
}

// Python 調用的封裝函數
torch::Tensor matmul_cuda(torch::Tensor x, torch::Tensor w) {
    const int M = x.size(0);
    const int N = x.size(1);
    const int K = w.size(1);

    auto y = torch::zeros({M, K}, torch::device(x.device()).dtype(x.dtype()));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((K + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_mul_cuda<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(), M, N, K
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda", &matmul_cuda, "CUDA 矩陣相乘");
}
