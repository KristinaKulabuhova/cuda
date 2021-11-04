#include <MatrixVectorMul.cuh>

__global__
void MatrixVectorMul(int height, int width, float* matrix, float* vector, float* result) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int stride_x = blockDim.x * gridDim.x;
	const int stride_y = blockDim.y * gridDim.y;

    for (int row = y; row < height; row += stride_row) {
        float* A_y = matrix + row * width;
        for(int col = x; col < width; col += stride_col) {
            atomicAdd(&result[row], A_y[col] * vector[col]);
        }
	}
}

