nvcc -ptx kernel.cu -o kernel.ptx

g++ cuda_driver_api.cpp -o cuda_driver_api -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda

./cuda_driver_api
