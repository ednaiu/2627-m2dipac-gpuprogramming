// =============================================================================
//  TP4 -- Matrix transposition on the GPU: At = A^T
//
// GB/s, measured with nvcc124u1.
// n       blocks   horizontal   vertical   shared   in place   copy
// 256     4.4      28.3         33.6       55.1     53.7       47.7
// 512     4.6      52.4         74.1       148.1    143.2      153.4
// 1000    4.6      60.5         118.7      154.6    176.2      173.4
// 1024    4.6      59.7         88.3       165.2    198.6      179.9
// 2048    4.6      68.7         95.1       192.0    193.8      213.4
// 4096    4.5      79.9         110.7      199.3    192.0      222.9
// Shared-memory versions are closest to copy; blocks of one thread are slowest.
// Some measured rates exceed copy, especially for small matrices.
// Copy is a reference measurement, not a strict upper bound for every run.
//
//  A and At are N x N single-precision matrices, allocated as 1D arrays of
//  N*N floats (malloc on the CPU, cudaMalloc on the GPU), both row-major:
//
//      A(i, j)  is at  A[i * N + j]        At(i, j)  is at  At[i * N + j]
//
//  and At(i, j) = A(j, i). N is any positive integer: every kernel must work
//  whatever N (round the grid up, guard the threads outside the matrix).
//
//  A transposition does no arithmetic: its speed is the bandwidth it gets
//  from global memory. For each kernel the program prints the time and the
//  effective bandwidth, (bytes read + bytes written) / time = 8 N^2 / time.
//  The copy kernel (given) reads A and writes At = A with coalesced accesses
//  only: its bandwidth is the ceiling for a transposition.
//
//  EXERCISES -- one kernel each:
//    1. transpose_blocks     : N x N blocks of one thread, one element each
//    2. transpose_horizontal : blocks of BSXY x BSXY threads, one element
//                              per thread, the warps touching A horizontally
//    3. transpose_vertical   : the same blocks, the warps touching A
//                              vertically
//    4. transpose_shmem      : the same blocks; each block moves one
//                              BSXY x BSXY tile of A to its place in At
//                              through a padded tile in shared memory, A and
//                              At both touched horizontally
//    5. transpose_inplace    : in place, on A itself, without At: a block
//                              moves two mirror tiles at once through shared
//                              memory
//  For each kernel, write its body and its grid and block dimensions in
//  main() (the launches are written; as given, every kernel runs with 1
//  block of 1 thread, so the program compiles and runs from the start and
//  prints WRONG until the kernels are written). Then set n to 256, 512,
//  1024, 2048, 4096 in turn, rerun, and compare the bandwidth of the five
//  kernels with the copy.
//
//  Build:  nvcc -O2 -arch=native transpose.cu -o transpose
//  Run:    ./transpose          (the matrix sizes are set in main())
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
//  Error checking (wrap every CUDA runtime call and kernel launch)
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } } while (0)

// A kernel launch returns nothing: check the launch itself, then wait for it
// to finish and check what happened while it ran.
#define CUDA_CHECK_KERNEL() do { \
  CUDA_CHECK(cudaGetLastError()); \
  CUDA_CHECK(cudaDeviceSynchronize()); } while (0)

// ----------------------------------------------------------------------------
//  Block and tile size
// ----------------------------------------------------------------------------
#define BSXY 32   // side of the 2D blocks and of the shared tiles: a warp is one row of a block

// Test data: A(i, j) = i * N + j, all different, and exact in float up to
// N = 4096 (4096^2 = 2^24): an element moved to a wrong place is caught. The
// check is exact at any N: a transposition moves values, it computes nothing.
// Above 4096 the values are rounded; a correct kernel still passes (the
// reference is rounded the same way), but two elements may round to the same
// float, so a misplaced element can go unnoticed.
inline float aval(int n, int i, int j) { return (float)((long long)i * n + j); }

// =============================================================================
//  Given: copy At = A, the ceiling
// =============================================================================
//  Blocks of BSXY x BSXY threads, thread (tx, ty) of block (bx, by) copies
//  the element (by*BSXY + ty, bx*BSXY + tx): threadIdx.x runs along a row. A
//  warp reads 32 consecutive floats of A and writes 32 consecutive floats of
//  At: both coalesced. Grid: ceil(N / BSXY) x ceil(N / BSXY), guards.
// =============================================================================
__global__ void copy_matrix(int n, const float *A, float *At)
{
  int i = blockIdx.y * BSXY + threadIdx.y;
  int j = blockIdx.x * BSXY + threadIdx.x;
  if (i < n && j < n) { At[i * n + j] = A[i * n + j]; }
}

// =============================================================================
//  1. N x N blocks of one thread
// =============================================================================
//  Block (i, j) = (blockIdx.x, blockIdx.y) copies A(i, j) to At(j, i). The
//  grid has exactly N x N blocks: no guard.
// =============================================================================
__global__ void transpose_blocks(int n, const float *A, float *At)
{
  // TODO
  int i = blockIdx.x;
  int j = blockIdx.y;
  At[j * n + i] = A[i * n + j];
}

// =============================================================================
//  2. Blocks of BSXY x BSXY threads, warps horizontal on A
// =============================================================================
//  Thread (tx, ty) of block (bx, by) copies A(i, j), i = by*BSXY + ty,
//  j = bx*BSXY + tx, to At(j, i): threadIdx.x runs along a row of A. Grid:
//  ceil(N / BSXY) x ceil(N / BSXY), guard on i and j.
//  Which of the two accesses of a warp is coalesced?
//  The read from A: consecutive threads read consecutive floats.
//  The write to At is strided, not coalesced.
// =============================================================================
__global__ void transpose_horizontal(int n, const float *A, float *At)
{
  // TODO
  int i = blockIdx.y * BSXY + threadIdx.y;
  int j = blockIdx.x * BSXY + threadIdx.x;
  if (i < n && j < n) { At[j * n + i] = A[i * n + j]; }
}

// =============================================================================
//  3. Blocks of BSXY x BSXY threads, warps vertical on A
// =============================================================================
//  Thread (tx, ty) of block (bx, by) copies A(i, j), i = bx*BSXY + tx,
//  j = by*BSXY + ty, to At(j, i): threadIdx.x runs down a column of A.
//  Grid: ceil(N / BSXY) x ceil(N / BSXY), guard on i and j.
//  Which of the two accesses of a warp is coalesced now?
//  The write to At: consecutive threads write consecutive floats.
//  The read from A is strided, not coalesced.
// Without shared memory, making one access contiguous makes the other
// strided: a row of A becomes a column of At. Both cannot be coalesced.
// =============================================================================
__global__ void transpose_vertical(int n, const float *A, float *At)
{
  // TODO
  int i = blockIdx.x * BSXY + threadIdx.x;
  int j = blockIdx.y * BSXY + threadIdx.y;
  if (i < n && j < n) { At[j * n + i] = A[i * n + j]; }
}

// =============================================================================
//  4. Through a padded tile in shared memory
// =============================================================================
//  Block (bx, by) moves the tile of A at rows by*BSXY.. and columns bx*BSXY..
//  to the tile of At at rows bx*BSXY.. and columns by*BSXY.., through a
//  BSXY x BSXY tile in shared memory, in two steps:
//    - load:  thread (tx, ty) reads A(by*BSXY + ty, bx*BSXY + tx) into the
//             tile -- threadIdx.x along a row of A;
//    - store: thread (tx, ty) writes At(bx*BSXY + ty, by*BSXY + tx) --
//             threadIdx.x along a row of At -- taking the value it needs
//             from the tile.
//  A and At are both touched horizontally: both global accesses of a warp
//  are coalesced. How the tile is touched does not matter for coalescing,
//  but PAD it, [BSXY][BSXY + 1], against bank conflicts. Do not forget the
//  barrier. Grid: ceil(N / BSXY) x ceil(N / BSXY); N is any positive
//  integer.
// Thread (tx, ty) stores the value loaded by thread (ty, tx).
// A warp reads tile[tx][ty]: 32 banks with padding, one bank without it.
// Without padding this gives a 32-way bank conflict.
// =============================================================================
__global__ void transpose_shmem(int n, const float *A, float *At)
{
  // TODO
  __shared__ float tile[BSXY][BSXY + 1];
  int tx = threadIdx.x, ty = threadIdx.y;
  int i = blockIdx.y * BSXY + ty;
  int j = blockIdx.x * BSXY + tx;
  if (i < n && j < n) { tile[ty][tx] = A[i * n + j]; }
  __syncthreads();
  i = blockIdx.x * BSXY + ty;
  j = blockIdx.y * BSXY + tx;
  if (i < n && j < n) { At[i * n + j] = tile[tx][ty]; }
}

// =============================================================================
//  5. In place, through shared memory
// =============================================================================
//  A is replaced by A^T, with no second matrix. The tile at block position
//  (bx, by) of A must end up, transposed, at position (by, bx), and the tile
//  at (by, bx) must end up, transposed, at (bx, by): a block that handles
//  the two mirror tiles together can do it, since it holds both in shared
//  memory before it overwrites either. Block (bx, by) with bx > by loads its
//  tile and its mirror into two padded shared tiles, then writes each one,
//  transposed, at the other's place. The diagonal blocks (bx == by) are
//  their own mirror. The blocks below the diagonal (bx < by) have nothing
//  to do. Grid: ceil(N / BSXY) x ceil(N / BSXY); N is any positive integer.
// =============================================================================
__global__ void transpose_inplace(int n, float *A)
{
  // TODO
  if (blockIdx.x < blockIdx.y) { return; }
  __shared__ float tile[BSXY][BSXY + 1];
  __shared__ float mirror[BSXY][BSXY + 1];
  int tx = threadIdx.x, ty = threadIdx.y;
  int i = blockIdx.y * BSXY + ty;
  int j = blockIdx.x * BSXY + tx;
  int mi = blockIdx.x * BSXY + ty;
  int mj = blockIdx.y * BSXY + tx;
  if (i < n && j < n) { tile[ty][tx] = A[i * n + j]; }
  if (mi < n && mj < n) { mirror[ty][tx] = A[mi * n + mj]; }
  __syncthreads();
  if (i < n && j < n) { A[i * n + j] = mirror[tx][ty]; }
  if (blockIdx.x != blockIdx.y && mi < n && mj < n) {
    A[mi * n + mj] = tile[tx][ty];
  }
}

// =============================================================================
//  Host side: clock, verification
// =============================================================================
// Wall-clock time in milliseconds. A kernel launch returns at once: call
// cudaDeviceSynchronize() (CUDA_CHECK_KERNEL does) before reading the clock.
double now(void)
{
  using namespace std::chrono;
  return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

// Compare every element of R with A^T (or with A, for the copy), A being
// known by its formula. Returns the number of wrong elements and prints the
// first one.
int verify(int n, const float *R, bool transposed)
{
  int errors = 0, firstI = -1, firstJ = -1;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float ref = transposed ? aval(n, j, i) : aval(n, i, j);
      if (R[i * n + j] != ref) {
        if (errors == 0) { firstI = i; firstJ = j; }
        errors++;
      }
    }
  }
  if (errors) {
    float ref = transposed ? aval(n, firstJ, firstI) : aval(n, firstI, firstJ);
    printf("      first wrong element: (%d, %d) = %g, expected %g\n",
           firstI, firstJ, R[firstI * n + firstJ], ref);
  }
  return errors;
}

// After a kernel: copy its result dR back into R, check it, print one line
// (result, time in ms, effective bandwidth in GB/s), then clear dR.
void check(const char *name, int n, double ms, bool transposed, float *R, float *dR)
{
  size_t bytes = (size_t)n * n * sizeof(float);
  CUDA_CHECK(cudaMemcpy(R, dR, bytes, cudaMemcpyDeviceToHost));
  int errors = verify(n, R, transposed);
  double gbs = 2.0 * bytes / (ms * 1e-3) / 1e9;   // bytes read + bytes written
  if (errors) { printf("%-30s %6d  WRONG (%d)\n", name, n, errors); }
  else        { printf("%-30s %6d  ok      %10.3f ms %10.1f GB/s\n", name, n, ms, gbs); }
  CUDA_CHECK(cudaMemset(dR, 0, bytes));
}

// =============================================================================
//  Driver
// =============================================================================
int main(void)
{
  const int sizes[] = {256, 512, 1000, 1024, 2048, 4096};
  for (int n : sizes) {
    size_t bytes = (size_t)n * n * sizeof(float);
    // One host array: A is written into it and sent to the GPU, then every
    // result is copied back into it and checked against the formula of A.
    float *R = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) { R[i * n + j] = aval(n, i, j); }
    }
    float *dA, *dAt;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dAt, bytes));
    CUDA_CHECK(cudaMemcpy(dA, R, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dAt, 0, bytes));
    printf("%-30s %6s  %-7s %13s %15s\n", "kernel", "N", "result", "time", "bandwidth");
    double t0;
    // Every kernel is launched twice. The first launch loads the kernel onto
    // the GPU (and, for the first kernel, finishes setting the device up) and
    // is not timed; the clock is read around the second one.

    // ---- given: copy, the ceiling -------------------------------------------
    dim3 grid0((n + BSXY - 1) / BSXY, (n + BSXY - 1) / BSXY), block0(BSXY, BSXY);
    copy_matrix<<<grid0, block0>>>(n, dA, dAt);   // warm-up
    CUDA_CHECK_KERNEL();
    t0 = now();
    copy_matrix<<<grid0, block0>>>(n, dA, dAt);   // timed
    CUDA_CHECK_KERNEL();
    check("copy (given)", n, now() - t0, false, R, dAt);

    // ---- 1. N x N blocks of one thread --------------------------------------
    // TODO: grid and block of kernel 1 (as given: 1 block of 1 thread)
    dim3 grid1(n, n), block1(1);
    transpose_blocks<<<grid1, block1>>>(n, dA, dAt);   // warm-up
    CUDA_CHECK_KERNEL();
    t0 = now();
    transpose_blocks<<<grid1, block1>>>(n, dA, dAt);   // timed
    CUDA_CHECK_KERNEL();
    check("1 blocks of one thread", n, now() - t0, true, R, dAt);

    // ---- 2. Warps horizontal on A -------------------------------------------
    // TODO: grid and block of kernel 2 (as given: 1 block of 1 thread)
    dim3 grid2 = grid0, block2 = block0;
    transpose_horizontal<<<grid2, block2>>>(n, dA, dAt);   // warm-up
    CUDA_CHECK_KERNEL();
    t0 = now();
    transpose_horizontal<<<grid2, block2>>>(n, dA, dAt);   // timed
    CUDA_CHECK_KERNEL();
    check("2 warps horizontal on A", n, now() - t0, true, R, dAt);

    // ---- 3. Warps vertical on A ---------------------------------------------
    // TODO: grid and block of kernel 3 (as given: 1 block of 1 thread)
    dim3 grid3 = grid0, block3 = block0;
    transpose_vertical<<<grid3, block3>>>(n, dA, dAt);   // warm-up
    CUDA_CHECK_KERNEL();
    t0 = now();
    transpose_vertical<<<grid3, block3>>>(n, dA, dAt);   // timed
    CUDA_CHECK_KERNEL();
    check("3 warps vertical on A", n, now() - t0, true, R, dAt);

    // ---- 4. Through a padded shared tile ------------------------------------
    // TODO: grid and block of kernel 4 (as given: 1 block of 1 thread)
    dim3 grid4 = grid0, block4 = block0;
    transpose_shmem<<<grid4, block4>>>(n, dA, dAt);   // warm-up
    CUDA_CHECK_KERNEL();
    t0 = now();
    transpose_shmem<<<grid4, block4>>>(n, dA, dAt);   // timed
    CUDA_CHECK_KERNEL();
    check("4 padded shared tile", n, now() - t0, true, R, dAt);

    // ---- 5. In place --------------------------------------------------------
    // The kernel transposes the matrix it is given: dAt receives a copy of A
    // before each launch, and is checked afterwards.
    // TODO: grid and block of kernel 5 (as given: 1 block of 1 thread)
    dim3 grid5 = grid0, block5 = block0;
    CUDA_CHECK(cudaMemcpy(dAt, dA, bytes, cudaMemcpyDeviceToDevice));
    transpose_inplace<<<grid5, block5>>>(n, dAt);   // warm-up
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(dAt, dA, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());            // the copy is finished before the clock starts
    t0 = now();
    transpose_inplace<<<grid5, block5>>>(n, dAt);   // timed
    CUDA_CHECK_KERNEL();
    check("5 in place", n, now() - t0, true, R, dAt);

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dAt));
    free(R);
    printf("\n");
  }
  return 0;
}
