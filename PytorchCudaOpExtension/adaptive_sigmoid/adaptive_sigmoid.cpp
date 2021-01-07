#include <torch/extension.h>
#include "adaptive_sigmoid.h"

at::Tensor adaptive_sigmoid_forward(
    at::Tensor input,
    at::Tensor params
){
    int batch = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto output = at::empty({batch, channels, height, width}, input.options());
    
    auto input_ptr = input.data<float>();
    auto output_ptr = output.data<float>();
    auto params_ptr = params.data<float>();
    
    for(int i = 0; i<batch; i++){
        auto input_instance_ptr = input_ptr + i * channels * height * width;
        auto output_instance_ptr = output_ptr + i * channels * height * width;
        adaptive_sigmoid_fucntion(
            THCState_getCurrentStream(state),
            input_instance_ptr,
            params_ptr,
            output_instance_ptr,
            channels, height, width
        );
    }
    
    return output;
}

std::vector<at::Tensor> adaptive_sigmoid_backward(
    at::Tensor input,
    at::Tensor params,
    at::Tensor grad_outputs,
    bool alpha_update,
    bool beta_update,
    bool gamma_update,
    bool theta_update
){
    int batch = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto grad_input = at::zeros_like(input);
    auto grad_params = at::zeros_like(params);
    
    auto input_ptr = input.data<float>();
    auto grad_output_ptr = grad_outputs.data<float>();
    auto params_ptr = params.data<float>();
    auto grad_input_ptr = grad_input.data<float>();
    auto grad_params_ptr = grad_params.data<float>();
    
    for(int i = 0; i < batch; i++){
        auto input_instance_ptr = input_ptr + i * channels * height * width;
        auto grad_output_instance_ptr = grad_output_ptr + i * channels * height * width;
        auto grad_input_instance_ptr = grad_input_ptr + i * channels * height * width;
        adaptive_sigmoid_input_grad(
            THCState_getCurrentStream(state),
            input_instance_ptr,
            grad_output_instance_ptr,
            params_ptr,
            grad_input_instance_ptr,
            channels, height, width
        );
        
        adaptive_sigmoid_params_grad(
            THCState_getCurrentStream(state),
            input_instance_ptr,
            grad_output_instance_ptr,
            params_ptr,
            grad_params_ptr,
            channels, height, width,
            alpha_update, beta_update, gamma_update, theta_update
        );
    }
    
    return {grad_input, grad_params};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("forward", &adaptive_sigmoid_forward, "adaptive sigmoid forward (CUDA)");
  m.def("backward", &adaptive_sigmoid_backward, "adaptive sigmoid backward (CUDA)");
}