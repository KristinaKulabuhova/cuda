#include <KernelMatrixAdd.cuh>

__global__ void KernelMatrixAdd(int height, int width, int pitch, float* A, float* B, float* result) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int stride_x = blockDim.x * gridDim.x;
	const int stride_y = blockDim.y * gridDim.y;

  for (int row = y; row < height; row += stride_y) {
    float* A_y = reinterpret_cast<float*>(reinterpret_cast<char*>(A) + row * pitch);
    float* B_y = reinterpret_cast<float*>(reinterpret_cast<char*>(B) + row * pitch);
    float* res_y = reinterpret_cast<float*>(reinterpret_cast<char*>(result) + row * pitch);

    for(int col = x; col < width; col += stride_x) {
      res_y[col] = A_y[col] + B_y[col];
    }
	}
}

