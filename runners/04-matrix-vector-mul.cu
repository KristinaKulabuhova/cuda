#include <cassert>
#include <iostream>

#include <MatrixVectorMul.cuh>

const float TOLERANCE = 0.001f;

void FillMatrix(float* mat, int W, int H, float value) {
}

int main() {
  const int W = 10000;
  const int H = 10000;

  dim3 block_size(32, 32);
  dim3 n_blocks((W + block_size.x - 1) / block_size.x, (H + block_size.y - 1) / block_size.y);

  float *h_A = new float[W * H];
  float *h_v = new float[W];
  float *h_result = new float[W];

  for(int row = 0; row < H; ++row) {
    for(int col = 0; col < W; ++col) {
      h_A[row * W + col] = 1.0f;
    }
  }

  for (int row = 0; row < W; ++row) {
    h_v[row] = 1.0f;
  }

  float *d_A, *d_v, *d_result;

  cudaMalloc(&d_A, W * H * sizeof(float));
  cudaMalloc(&d_v, W * sizeof(float));
  cudaMalloc(&d_result, W * sizeof(float));

  cudaMemcpy(d_A, h_A, W * H * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v, W * sizeof(float), cudaMemcpyHostToDevice);

  MatrixVectorMul<<<n_blocks, block_size>>>(H, W, d_A, d_v, d_result);
  cudaDeviceSynchronize();

  cudaMemcpy(h_result, d_result, W * sizeof(float), cudaMemcpyDeviceToHost);

  for (int row = 0; row < W; ++row) {
    assert(h_result[row] - 10000.0f <= 0.01f);
  }

  cudaFree(d_A);
  cudaFree(d_v);
  cudaFree(d_result);

  delete[] h_result;
  delete[] h_v;
  delete[] h_A;

  return 0;
}
