#pragma once
#include "simulator.hpp"
namespace sjtu {

// Helper to vertically stack rows [0..i] into a single (i+1, d) matrix in SRAM.
static Matrix *BuildStacked(MatrixMemoryAllocator &allocator, GpuSimulator &gpu_sim,
                            const std::vector<Matrix *> &mats, size_t upto_inclusive) {
  assert(upto_inclusive < mats.size());
  // Ensure first row is in SRAM and copy to initialize the stack
  gpu_sim.MoveMatrixToSharedMem(mats[0]);
  auto stacked = allocator.Allocate("stack_init");
  gpu_sim.Copy(mats[0], stacked, Position::kInSharedMemory);
  // Append remaining rows one by one
  for (size_t j = 1; j <= upto_inclusive; ++j) {
    gpu_sim.MoveMatrixToSharedMem(mats[j]);
    auto new_stacked = allocator.Allocate("stack_concat");
    gpu_sim.Concat(stacked, mats[j], new_stacked, /*axis=*/0, Position::kInSharedMemory);
    // Release previous partial stack after concatenation
    gpu_sim.ReleaseMatrix(stacked);
    stacked = new_stacked;
  }
  return stacked;
}

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    // Get Q for this round: shape (i+1, d), initially in HBM
    Matrix *Q = rater.GetNextQuery();

    // Move Q to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(Q);

    // Streaming attention computation: build Y row-by-row without forming large intermediate matrices.
    // Ensure keys/values needed this round are in shared memory.
    for (size_t j = 0; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      gpu_sim.MoveMatrixToSharedMem(values[j]);
    }

    Matrix *Y = nullptr; // final answer (i+1, d)
    for (size_t r = 0; r <= i; ++r) {
      // Get Q row r in shared memory: (1, d)
      Matrix *Q_row = matrix_memory_allocator.Allocate("Q_row");
      gpu_sim.GetRow(Q, r, Q_row, Position::kInSharedMemory);

      // Numerator accumulator y_num (1, d), Denominator denom (1,1)
      Matrix *y_num = nullptr;
      Matrix *denom = nullptr;

      for (size_t j = 0; j <= i; ++j) {
        // Prepare K[j]^T as a column without modifying original keys[j]
        Matrix *K_col = matrix_memory_allocator.Allocate("K_col");
        gpu_sim.Copy(keys[j], K_col, Position::kInSharedMemory);
        gpu_sim.Transpose(K_col, Position::kInSharedMemory); // (d,1)

        // s_rj = Q_row (1,d) @ K_col (d,1) -> (1,1)
        Matrix *s_rj = matrix_memory_allocator.Allocate("s_rj");
        gpu_sim.MatMul(Q_row, K_col, s_rj);
        Matrix *e_rj = matrix_memory_allocator.Allocate("e_rj");
        gpu_sim.MatExp(s_rj, e_rj);

        // Accumulate denom
        if (j == 0) {
          denom = matrix_memory_allocator.Allocate("denom");
          gpu_sim.Copy(e_rj, denom, Position::kInSharedMemory);
        } else {
          Matrix *den_new = matrix_memory_allocator.Allocate("den_new");
          gpu_sim.MatAdd(denom, e_rj, den_new);
          gpu_sim.ReleaseMatrix(denom);
          denom = den_new;
        }

        // Accumulate y_num += e_rj * V[j]
        Matrix *scaled_v = matrix_memory_allocator.Allocate("scaled_v");
        gpu_sim.MatMul(e_rj, values[j], scaled_v); // (1,1) x (1,d) -> (1,d)
        if (j == 0) {
          y_num = matrix_memory_allocator.Allocate("y_num");
          gpu_sim.Copy(scaled_v, y_num, Position::kInSharedMemory);
        } else {
          Matrix *y_new = matrix_memory_allocator.Allocate("y_new");
          gpu_sim.MatAdd(y_num, scaled_v, y_new);
          gpu_sim.ReleaseMatrix(y_num);
          y_num = y_new;
        }

        // Release small temporaries
        gpu_sim.ReleaseMatrix(K_col);
        gpu_sim.ReleaseMatrix(s_rj);
        gpu_sim.ReleaseMatrix(e_rj);
        gpu_sim.ReleaseMatrix(scaled_v);
      }

      // y_r = y_num / denom -> (1, d)
      Matrix *y_r = matrix_memory_allocator.Allocate("y_row");
      gpu_sim.MatDiv(y_num, denom, y_r);

      // Append to Y
      if (r == 0) {
        Y = matrix_memory_allocator.Allocate("answer");
        gpu_sim.Copy(y_r, Y, Position::kInSharedMemory);
      } else {
        Matrix *Y_new = matrix_memory_allocator.Allocate("Y_new");
        gpu_sim.Concat(Y, y_r, Y_new, /*axis=*/0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(Y);
        Y = Y_new;
      }

      // Release per-row accumulators
      gpu_sim.ReleaseMatrix(Q_row);
      gpu_sim.ReleaseMatrix(y_num);
      gpu_sim.ReleaseMatrix(denom);
      gpu_sim.ReleaseMatrix(y_r);
    }

    // Move answer to HBM before committing
    gpu_sim.MoveMatrixToGpuHbm(Y);

    // Run queued instructions, then commit the answer from HBM
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*Y);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
