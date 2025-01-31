nvcc -ptx kernel.cu -o kernel.ptx
g++ ptx_launcher.cpp -o ptx_launcher -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda
./ptx_launcher
