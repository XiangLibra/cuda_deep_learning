extern "C" __global__ void my_kernel(float *data) {
    int idx = threadIdx.x;
    data[idx] = data[idx] * 2.0f;  // 簡單的計算：每個值乘 2
}
