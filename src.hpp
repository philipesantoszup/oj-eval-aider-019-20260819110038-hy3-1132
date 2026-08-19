#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    // current_query is in HBM, shape [i+1, 512]

    // 1. Build K_concat and V_concat in HBM (using copies to avoid modifying
    // original keys/values)
    Matrix *K_concat = matrix_memory_allocator.Allocate("K_concat");
    gpu_sim.Copy(keys[0], K_concat, kInGpuHbm);
    for (size_t j = 1; j <= i; ++j) {
      Matrix *new_K = matrix_memory_allocator.Allocate("K_concat_tmp");
      gpu_sim.Concat(K_concat, keys[j], new_K, 0, kInGpuHbm);
      gpu_sim.ReleaseMatrix(K_concat);
      K_concat = new_K;
    }
    Matrix *V_concat = matrix_memory_allocator.Allocate("V_concat");
    gpu_sim.Copy(values[0], V_concat, kInGpuHbm);
    for (size_t j = 1; j <= i; ++j) {
      Matrix *new_V = matrix_memory_allocator.Allocate("V_concat_tmp");
      gpu_sim.Concat(V_concat, values[j], new_V, 0, kInGpuHbm);
      gpu_sim.ReleaseMatrix(V_concat);
      V_concat = new_V;
    }

    // 2. Move Q, K_concat, V_concat to SRAM for calculation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K_concat);
    gpu_sim.MoveMatrixToSharedMem(V_concat);

    // 3. Transpose K_concat in SRAM to obtain K^T (shape [d, i+1])
    gpu_sim.Transpose(K_concat, kInSharedMemory);

    // 4. Compute attention row by row
    Matrix *answer_concat = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *q_row = matrix_memory_allocator.Allocate("q_row");
      gpu_sim.GetRow(current_query, r, q_row, kInSharedMemory);

      Matrix *scores = matrix_memory_allocator.Allocate("scores");
      gpu_sim.MatMul(q_row, K_concat, scores);

      Matrix *exp_scores = matrix_memory_allocator.Allocate("exp_scores");
      gpu_sim.MatExp(scores, exp_scores);

      Matrix *sum_exp = matrix_memory_allocator.Allocate("sum_exp");
      gpu_sim.Sum(exp_scores, sum_exp);

      Matrix *softmax = matrix_memory_allocator.Allocate("softmax");
      gpu_sim.MatDiv(exp_scores, sum_exp, softmax);

      Matrix *out_row = matrix_memory_allocator.Allocate("out_row");
      gpu_sim.MatMul(softmax, V_concat, out_row);

      if (r == 0) {
        answer_concat = out_row;
      } else {
        Matrix *new_answer = matrix_memory_allocator.Allocate("answer_concat");
        gpu_sim.Concat(answer_concat, out_row, new_answer, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(answer_concat);
        gpu_sim.ReleaseMatrix(out_row);
        answer_concat = new_answer;
      }

      // Release intermediate matrices that are no longer needed
      gpu_sim.ReleaseMatrix(q_row);
      gpu_sim.ReleaseMatrix(scores);
      gpu_sim.ReleaseMatrix(exp_scores);
      gpu_sim.ReleaseMatrix(sum_exp);
      gpu_sim.ReleaseMatrix(softmax);
    }

    // 5. Release K_concat, V_concat, current_query (they are in SRAM and no
    // longer needed)
    gpu_sim.ReleaseMatrix(K_concat);
    gpu_sim.ReleaseMatrix(V_concat);
    gpu_sim.ReleaseMatrix(current_query);

    // 6. Move the final answer to HBM
    gpu_sim.MoveMatrixToGpuHbm(answer_concat);

    // 7. Execute all queued instructions
    gpu_sim.Run(false, &matrix_memory_allocator);

    // 8. Commit the answer (must be in HBM)
    rater.CommitAnswer(*answer_concat);
    // answer_concat is released automatically by CommitAnswer
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
