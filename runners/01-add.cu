#include <cstdint>
#include <iostream>

using std::cout;
using std::endl;

#include "KernelAdd.cuh"


int main() {
  const int N = (1 << 28);
  const int block_size = 256;
  const int n_blocks = (N + block_size - 1) / block_size;
  float *x, *y, *sum;
  const uint64_t array_byte_len = N * sizeof(*x);
  float time;

	cudaMallocManaged(&x, array_byte_len);
	cudaMallocManaged(&y, array_byte_len);
	cudaMallocManaged(&sum, array_byte_len);

	for (int i = 0; i < N; ++i) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  KernelAdd<<<n_blocks, block_size>>>(numElements, x, y, sum);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
  cudaEventElapsedTime(&time, start, stop);
  std::cout << time << std::endl;

  for(int i = 0; i < N; ++i) {
    assert(sum[i] == 3.0f);
  }

  cudaFree(x);
  cudaFree(y);
  cudaFree(sum);

  return 0;
}
