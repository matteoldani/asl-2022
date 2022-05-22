#include <optimizations/optimizations_utils.h>
myInt64 nnm_cost_2(int V_row, int V_col, int W_row, int W_col, int H_row, int H_col, int num_iterations){

    return (myInt64)(2 * W_row * H_col * W_col + 5 * V_row * V_col + 3) +
           (myInt64)num_iterations * (myInt64) (2 * W_row * H_col * W_col + 5 * V_row * V_col +
                            2 * W_col * V_col * V_row +
                            2 * W_col * W_col * W_row +
                            2 * W_col * W_col * H_col +
                            2 * V_row * H_row * V_col +
                            1 * H_row * H_col * H_col +
                            1 * H_row * H_row * W_row +
                            2 * H_row * H_col +
                            2 * W_row * W_col + 3);

}