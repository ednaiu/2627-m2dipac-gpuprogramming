#include <cstdio>
#include "cuda.h"

__global__ void cudaHello(){
  // Print "Hello World" together with blockIdx.x and threadIdx.x from every thread
  // A FAIRE ...
  // In the CUDA kernel, print the block id and thread id of each thread, as well as the total number of blocks
  // and threads/block (see the “Hello World, revisited” slide of the lecture for the four built-in variables)
  printf("Hello World from block %d, thread %d (total blocks: %d, threads/block: %d)\n", blockIdx.x, threadIdx.x, gridDim.x, blockDim.x);
}

int main() {
  int numBlocks = 1;
  int blockSize;

  // Experiment with different blockSize values (threads per block) over the powers of 2,
  // keeping the total number of threads equal to 64
  // A FAIRE ...
  for (numBlocks = 1; numBlocks <= 64; numBlocks *= 2) {
    blockSize = 64 / numBlocks;
    cudaHello<<<numBlocks, blockSize>>>();

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
      printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
  }
  return 0;
}
