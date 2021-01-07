#ifndef ADAPTIVE_SIGMOID
#define ADAPTIVE_SIGMOID
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

extern THCState *state;
typedef std::vector<int> TShape;

void adaptive_sigmoid_fucntion(
    cudaStream_t stream,
    const float* data_in,
    const float* params,
    float* output,
    int channels, int height, int width
);

void adaptive_sigmoid_input_grad(
    cudaStream_t stream,
    const float* data_in,
    const float* grad_outputs,
    const float* params,
    float* grad_input,
    int channels, int height, int width
);

void adaptive_sigmoid_params_grad(
    cudaStream_t stream,
    const float* data_in,
    const float* grad_outputs,
    const float* params,
    float* grad_params,
    int channels, int height, int width,
    bool alpha_update, 
    bool beta_update,
    bool gamma_update,
    bool theta_update
);

#endif