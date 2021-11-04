#include <assert.h>
#include <iostream>

#include <KernelMatrixAdd.cuh>

int main() {
  float *d_A, *d_B, *d_C;
  size_t pitch;
  int W = 10000;
  int H = 10000;

  dim3 block_size{32, 32}; // max threads in block 1024
  dim3 n_blocks{(W + block_size.x - 1) / block_size.x, (H + block_size.y - 1) / block_size.y};

  float *h_A = new float[W * H];
  float *h_B = new float[W * H];
  float *h_C = new float[W * H];

  for(int row = 0; row < H; ++row) {
    for(int col = 0; col < W; ++col) {
      h_A[row * W + col] = 1.0f;
      h_B[row * W + col] = 2.0f;
    }
  }

  cudaMallocPitch(&d_A, &pitch, W * sizeof(float), H);
  cudaMallocPitch(&d_B, &pitch, W * sizeof(float), H);
  cudaMallocPitch(&d_C, &pitch, W * sizeof(float), H);

  cudaMemcpy2D(d_A, pitch, h_A, W * sizeof(float), W * sizeof(float), H, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_B, pitch, h_B, W * sizeof(float), W * sizeof(float), H, cudaMemcpyHostToDevice);

  KernelMatrixAdd<<<n_blocks, block_size>>>(H, W, pitch, d_A, d_B, d_C);
	cudaDeviceSynchronize();

  cudaMemcpy2D(h_C, W * sizeof(float), d_C, pitch, W * sizeof(float), H, cudaMemcpyDeviceToHost);

  for (int row = 0; row < H; ++row) {
    for(int col = 0; col < W; ++col) {
      assert(h_C[row * W + col] == 3.0f);
    }
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}