#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include "adaptive_sigmoid.h"

#define CUDA_KERNEL_LOOP(i ,n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<(n); i+= blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N){
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void adaptive_sigmoid_fucntion_kernel(
    int n,
    const float* data_in,
    const float* params,
    float* output
){
    CUDA_KERNEL_LOOP(index, n){
        float alpha = params[0];
        float beta = params[1];
        float gamma = params[2];
        float theta = params[3];
        float value = data_in[index];
 //       output[index] = gamma * (1 / (1 + exp(-alpha * (value - beta)))) + theta;
        output[index] = gamma * (1 / (1 + exp(-alpha * (value - beta))) - theta);
    }
}

__global__ void adaptive_sigmoid_input_grad_kernel(
    int n,
    const float* data_in,
    const float* grad_output,
    const float* params,
    float* grad_input
){
    CUDA_KERNEL_LOOP(index, n){
        float alpha = params[0];
        float beta = params[1];
        float gamma = params[2];
        float value = data_in[index];
        float d_grad_output = grad_output[index];
        float efx = exp(- alpha * (value - beta));
        float patial = efx / ((1 + efx) * (1 + efx));
        grad_input[index] = gamma * alpha * patial * d_grad_output;
    }
}

__global__ void adaptive_sigmoid_params_grad_kernel(
    int n,
    const float* data_in,
    const float* grad_output,
    const float* params,
    float* grad_params,
    bool alpha_update, 
    bool beta_update,
    bool gamma_update,
    bool theta_update
){
    CUDA_KERNEL_LOOP(index, n){
        float alpha = params[0];
        float beta = params[1];
        float gamma = params[2];
        float value = data_in[index];
        float d_grad_output = grad_output[index];
        float efx = exp(- alpha * (value - beta));
        float patial = efx / ((1 + efx) * (1 + efx));
        
        float d_alpha = gamma * patial * (value - beta);
        float d_beta = gamma * patial * (- alpha);
        float d_gamma = 1 / (1 + efx);
        float d_theta = -gamma;
        // float d_beta = 0;
        // float d_gamma = 0;
        // float d_theta = 0;
        if (alpha_update)
            atomicAdd(grad_params + 0, d_alpha * d_grad_output);
        if (beta_update)
            atomicAdd(grad_params + 1, d_beta * d_grad_output);
        if (gamma_update)
            atomicAdd(grad_params + 2, d_gamma * d_grad_output);
        if (theta_update)
            atomicAdd(grad_params + 3, d_theta * d_grad_output);
    }
}

void adaptive_sigmoid_fucntion(
    cudaStream_t stream,
    const float* data_in,
    const float* params,
    float* output,
    int channels, int height, int width
){
    int num_kernels = channels * height * width;
    adaptive_sigmoid_fucntion_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        data_in,
        params,
        output
    );
}

void adaptive_sigmoid_input_grad(
    cudaStream_t stream,
    const float* data_in,
    const float* grad_outputs,
    const float* params,
    float* grad_input,
    int channels, int height, int width
){
    int num_kernels = channels * height * width;
    adaptive_sigmoid_input_grad_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        data_in,
        grad_outputs,
        params,
        grad_input
    );
}

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
){
    int num_kernels = channels * height * width;
    adaptive_sigmoid_params_grad_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        data_in,
        grad_outputs,
        params,
        grad_params,
        alpha_update, 
        beta_update,
        gamma_update,
        theta_update
    );
}