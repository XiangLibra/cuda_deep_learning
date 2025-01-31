#include <cuda.h>
#include <iostream>

#define CHECK_CUDA(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char *errStr; \
            cuGetErrorString(err, &errStr); \
            std::cerr << "CUDA ERROR: " << errStr << " at " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

int main() {
    // ✅ 初始化 CUDA
    CHECK_CUDA(cuInit(0));

    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    // ✅ 加載 PTX 模組
    CUmodule module;
    CHECK_CUDA(cuModuleLoad(&module, "kernel.ptx"));

    // ✅ 取得 CUDA 內核函數
    CUfunction kernel;
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, "vector_add"));

    // ✅ 分配裝置記憶體
    int N = 10;
    float h_A[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float h_B[10] = {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float h_C[10] = {0};

    CUdeviceptr d_A, d_B, d_C;
    CHECK_CUDA(cuMemAlloc(&d_A, N * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_B, N * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_C, N * sizeof(float)));

    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, N * sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(d_B, h_B, N * sizeof(float)));

    // ✅ 設定 Kernel 參數
    void *args[] = { &d_A, &d_B, &d_C, &N };

    // ✅ 執行 CUDA 內核
    CHECK_CUDA(cuLaunchKernel(kernel, 1, 1, 1, N, 1, 1, 0, 0, args, 0));

    // ✅ 拷貝結果回 Host
    CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, N * sizeof(float)));

    // ✅ 輸出結果
    std::cout << "結果: ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // ✅ 清理資源
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
