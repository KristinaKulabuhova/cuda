#include <KernelMul.cuh>

__global__ void KernelMul(int numElements, float* x, float* y, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for (int i = idx; i < numElements; i += stride) {
      result[i] = x[i] + y[i];
  }
}

