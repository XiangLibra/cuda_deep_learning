extern "C" __global__ void vector_add(float *A, float *B, float *C, int N) {
    int idx = threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
