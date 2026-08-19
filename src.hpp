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

    // Move query to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Create SRAM copies of keys (transposed) and values for this round
    std::vector<Matrix *> sram_keys;
    std::vector<Matrix *> sram_values;
    for (size_t j = 0; j <= i; ++j) {
      Matrix *k_copy = matrix_memory_allocator.Allocate("key_copy");
      gpu_sim.Copy(keys[j], k_copy, kInGpuHbm);
      gpu_sim.Run(false, &matrix_memory_allocator); // ensure copy completes (k_copy in HBM)
      gpu_sim.MoveMatrixToSharedMem(k_copy);
      gpu_sim.Run(false, &matrix_memory_allocator); // ensure move completes (k_copy in SRAM)
      gpu_sim.Transpose(k_copy, kInSharedMemory);
      sram_keys.push_back(k_copy);

      Matrix *v_copy = matrix_memory_allocator.Allocate("value_copy");
      gpu_sim.Copy(values[j], v_copy, kInGpuHbm);
      gpu_sim.Run(false, &matrix_memory_allocator);
      gpu_sim.MoveMatrixToSharedMem(v_copy);
      gpu_sim.Run(false, &matrix_memory_allocator);
      sram_values.push_back(v_copy);
    }

    Matrix *answer_concat = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *q_row = matrix_memory_allocator.Allocate("q_row");
      gpu_sim.GetRow(current_query, r, q_row, kInSharedMemory);

      // Compute scores by dot product with each key
      Matrix *scores = nullptr;
      for (size_t j = 0; j <= i; ++j) {
        Matrix *score_j = matrix_memory_allocator.Allocate("score_j");
        gpu_sim.MatMul(q_row, sram_keys[j], score_j);
        if (j == 0) {
          scores = score_j;
        } else {
          Matrix *new_scores = matrix_memory_allocator.Allocate("scores");
          gpu_sim.Concat(scores, score_j, new_scores, 1, kInSharedMemory);
          gpu_sim.ReleaseMatrix(scores);
          gpu_sim.ReleaseMatrix(score_j);
          scores = new_scores;
        }
      }

      Matrix *exp_scores = matrix_memory_allocator.Allocate("exp_scores");
      gpu_sim.MatExp(scores, exp_scores);
      Matrix *sum_exp = matrix_memory_allocator.Allocate("sum_exp");
      gpu_sim.Sum(exp_scores, sum_exp);
      Matrix *softmax = matrix_memory_allocator.Allocate("softmax");
      gpu_sim.MatDiv(exp_scores, sum_exp, softmax);

      // Compute out_row = sum_j softmax[j] * value_j
      Matrix *out_row = nullptr;
      for (size_t j = 0; j <= i; ++j) {
        Matrix *softmax_j = matrix_memory_allocator.Allocate("softmax_j");
        gpu_sim.GetColumn(softmax, j, softmax_j, kInSharedMemory);
        Matrix *scaled_v = matrix_memory_allocator.Allocate("scaled_v");
        gpu_sim.MatMul(softmax_j, sram_values[j], scaled_v);
        if (j == 0) {
          out_row = scaled_v;
        } else {
          Matrix *new_out = matrix_memory_allocator.Allocate("out_row");
          gpu_sim.MatAdd(out_row, scaled_v, new_out);
          gpu_sim.ReleaseMatrix(out_row);
          gpu_sim.ReleaseMatrix(scaled_v);
          out_row = new_out;
        }
        gpu_sim.ReleaseMatrix(softmax_j);
      }

      if (r == 0) {
        answer_concat = out_row;
      } else {
        Matrix *new_answer = matrix_memory_allocator.Allocate("answer_concat");
        gpu_sim.Concat(answer_concat, out_row, new_answer, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(answer_concat);
        gpu_sim.ReleaseMatrix(out_row);
        answer_concat = new_answer;
      }

      // Release intermediate matrices for this row
      gpu_sim.ReleaseMatrix(q_row);
      gpu_sim.ReleaseMatrix(scores);
      gpu_sim.ReleaseMatrix(exp_scores);
      gpu_sim.ReleaseMatrix(sum_exp);
      gpu_sim.ReleaseMatrix(softmax);
    }

    // Release SRAM copies of keys and values
    for (size_t j = 0; j <= i; ++j) {
      gpu_sim.ReleaseMatrix(sram_keys[j]);
      gpu_sim.ReleaseMatrix(sram_values[j]);
    }
    gpu_sim.ReleaseMatrix(current_query);

    // Move final answer to HBM
    gpu_sim.MoveMatrixToGpuHbm(answer_concat);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer_concat);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
