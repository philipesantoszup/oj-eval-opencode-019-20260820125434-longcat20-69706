#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  size_t d = 512;

  Matrix *K_all = nullptr;
  Matrix *V_all = nullptr;

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t n = i + 1;

    if (i == 0) {
      // Move first key, value, query to SRAM
      gpu_sim.MoveMatrixToSharedMem(keys[0]);
      gpu_sim.MoveMatrixToSharedMem(values[0]);
      gpu_sim.MoveMatrixToSharedMem(current_query);

      // Create K_all and V_all as copies in SRAM
      K_all = matrix_memory_allocator.Allocate("K_all");
      V_all = matrix_memory_allocator.Allocate("V_all");
      gpu_sim.Copy(keys[0], K_all, kInSharedMemory);
      gpu_sim.Copy(values[0], V_all, kInSharedMemory);
    } else {
      // Move new key, value, query to SRAM
      gpu_sim.MoveMatrixToSharedMem(keys[i]);
      gpu_sim.MoveMatrixToSharedMem(values[i]);
      gpu_sim.MoveMatrixToSharedMem(current_query);

      // Concatenate to grow K_all and V_all in SRAM
      Matrix *new_K = matrix_memory_allocator.Allocate("K_new");
      Matrix *new_V = matrix_memory_allocator.Allocate("V_new");
      gpu_sim.Concat(K_all, keys[i], new_K, 0, kInSharedMemory);
      gpu_sim.Concat(V_all, values[i], new_V, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(K_all);
      gpu_sim.ReleaseMatrix(V_all);
      K_all = new_K;
      V_all = new_V;
    }

    // Copy K_all and transpose for K^T
    Matrix *K_copy = matrix_memory_allocator.Allocate("K_copy");
    gpu_sim.Copy(K_all, K_copy, kInSharedMemory);
    gpu_sim.Transpose(K_copy, kInSharedMemory);

    // scores = Q @ K^T  -> (n, n)
    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(current_query, K_copy, scores);
    gpu_sim.ReleaseMatrix(K_copy);

    // Compute softmax row by row
    Matrix *softmax = nullptr;
    for (size_t j = 0; j < n; ++j) {
      Matrix *row = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(scores, j, row, kInSharedMemory);

      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row");
      gpu_sim.MatExp(row, exp_row);
      gpu_sim.ReleaseMatrix(row);

      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(exp_row, row_sum);

      Matrix *softmax_row = matrix_memory_allocator.Allocate("softmax_row");
      gpu_sim.MatDiv(exp_row, row_sum, softmax_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(row_sum);

      if (j == 0) {
        softmax = softmax_row;
      } else {
        Matrix *new_softmax = matrix_memory_allocator.Allocate("softmax");
        gpu_sim.Concat(softmax, softmax_row, new_softmax, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax);
        gpu_sim.ReleaseMatrix(softmax_row);
        softmax = new_softmax;
      }
    }
    gpu_sim.ReleaseMatrix(scores);

    // output = softmax @ V  -> (n, d)
    Matrix *output = matrix_memory_allocator.Allocate("output");
    gpu_sim.MatMul(softmax, V_all, output);
    gpu_sim.ReleaseMatrix(softmax);

    // Move output to HBM
    gpu_sim.MoveMatrixToGpuHbm(output);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*output);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
