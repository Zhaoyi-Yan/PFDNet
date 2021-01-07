#include <torch/extension.h>
#include "pad_conv2d.h"



at::Tensor pad_conv2d_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor rate_map,
    at::Tensor bias,
    int stride_h, int stride_w
//     int dilation_h, int dilation_w,
//     int pad_h, int pad_w
){
    /**
    * get the input parameter's information
    **/
    int batch = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
//     int kernel_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int rate_map_height = rate_map.size(2);
    int rate_map_width = rate_map.size(3);
    int height_out = (input_height - (1 * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_out = (input_width - (1 * (kernel_w - 1) + 1)) / stride_w + 1;
    /**
    * data correctness validation
    **/
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(rate_map.type().is_cuda(), "rate_map must be a CUDA tensor");
    AT_ASSERTM(rate_map_height == height_out, "output height must be same with rate map height");
    AT_ASSERTM(rate_map_width == width_out, "output width must be same with rate map width");
    AT_ASSERTM(kernel_h % 2 == 1 || kernel_w % 2 ==1, "kernel_size must be odd number");
    /**
    * derive more information
    **/
    int kernel_dim = in_channels * kernel_h * kernel_w;
    int input_dim = in_channels * input_height * input_width;
    int conv_out_spatial_dim = height_out * width_out;

    int M = out_channels;
    int N = conv_out_spatial_dim;
    int K = kernel_dim;
    /**
    * malloc tmp space and output space
    **/
    auto col_buffer = at::empty({in_channels * kernel_h * kernel_w, conv_out_spatial_dim}, input.options());
    auto output = at::empty({batch, out_channels, height_out, width_out}, input.options());
    /**
    * get pointer of the tensors
    **/
    auto input_ptr = input.data<float>();
    auto weight_ptr = weight.data<float>();
    auto rate_map_ptr = rate_map.data<float>();
    auto col_buffer_ptr = col_buffer.data<float>();
    auto output_ptr = output.data<float>();
    auto bias_ptr = bias.data<float>();

    for (int n = 0; n < batch; ++n) {
        pad_conv2d_im2col(
            THCState_getCurrentStream(state),
            input_ptr + n * input_dim,
            rate_map_ptr + n * conv_out_spatial_dim,
            in_channels, input_height, input_width,
            kernel_h, kernel_w,
//             pad_h, pad_w,
            stride_h, stride_w,
//             dilation_h, dilation_w,
            height_out, width_out,
            col_buffer_ptr
        );
        auto output_instance_ptr = output_ptr + (n * M  * N);
        THCudaBlas_Sgemm(state, 'n', 'n', N, M, K, 1.0f, col_buffer_ptr, N, weight_ptr, K, 0.0f, output_instance_ptr, N);
        add_bias(
            THCState_getCurrentStream(state),
            output_instance_ptr, 
            bias_ptr, 
            out_channels, height_out, width_out
        );
    }
    return output;
}

std::vector<at::Tensor> pad_conv2d_backward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor rate_map,
    at::Tensor bias,
    at::Tensor out_grad,
    int stride_h, int stride_w
//     int dilation_h, int dilation_w,
//     int pad_h, int pad_w
){
    /**
    * get the input parameter's information
    **/
    int batch = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
//     int kernel_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int rate_map_height = rate_map.size(2);
    int rate_map_width = rate_map.size(3);
    int height_out = (input_height  - (1 * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_out = (input_width - (1 * (kernel_w - 1) + 1)) / stride_w + 1;
    /**
    * data correctness validation
    **/
    AT_ASSERTM(height_out==out_grad.size(2) && width_out == out_grad.size(3),
        "the calculated out shape won't match the out_grad_shape:(%d x %d vs %d x %d)",
            height_out, width_out, out_grad.size(2), out_grad.size(3));
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(rate_map.type().is_cuda(), "rate_map must be a CUDA tensor");
    AT_ASSERTM(rate_map_height == height_out, "output height must be same with rate map height");
    AT_ASSERTM(rate_map_width == width_out, "output width must be same with rate map width");
    /**
    * derive more information
    **/
    int kernel_dim = in_channels * kernel_h * kernel_w;
    int input_dim = in_channels * input_height * input_width;
    int conv_out_spatial_dim = height_out * width_out;

    int M = kernel_dim;
    int N = conv_out_spatial_dim;
    int K = out_channels;
    /**
    * malloc tmp space and output space
    **/
    auto col_buffer = at::empty({in_channels * kernel_h * kernel_w,  conv_out_spatial_dim}, input.options());
    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros_like(weight);
    auto grad_bias = at::zeros_like(bias);
    auto grad_rate_map = at::zeros_like(rate_map);
    /**
    * get pointer of the tensors
    **/
    auto input_ptr = input.data<float>();
    auto weight_ptr = weight.data<float>();
    auto rate_map_ptr = rate_map.data<float>();
    auto out_grad_ptr = out_grad.data<float>();
    auto col_buffer_ptr = col_buffer.data<float>();
    auto grad_input_ptr = grad_input.data<float>();
    auto grad_weight_ptr = grad_weight.data<float>();
    auto grad_bias_ptr = grad_bias.data<float>();
//     auto bias_ptr = bias.data<float>();
    auto grad_rate_map_ptr = grad_rate_map.data<float>();
    
    for (int n = 0; n < batch; ++n) {
        auto out_grad_instance_ptr = out_grad_ptr + n * K * N;
        calculate_dbias(
            THCState_getCurrentStream(state),
            out_grad_instance_ptr,
            grad_bias_ptr,
            out_channels,
            height_out, width_out
            );
        THCudaBlas_Sgemm(state,
            'n', 't',
            N, M, K,
            1.0f,
            out_grad_instance_ptr, N,
            weight_ptr, M,
            0.0f,
            col_buffer_ptr, N);
        /**
        * calculate d loss / d rate_map
        **/
       pad_conv2d_col2im_coord(
           THCState_getCurrentStream(state),
           col_buffer_ptr,
           input_ptr + n * input_dim,
           rate_map_ptr + n * conv_out_spatial_dim,
           in_channels, input_height, input_width,
           kernel_h, kernel_w,
           stride_h, stride_w,
           height_out, width_out,
           grad_rate_map_ptr + n * conv_out_spatial_dim
       );

        /**
        * calculate d loss / d input
        **/
        pad_conv2d_col2im(
            THCState_getCurrentStream(state),
            col_buffer_ptr,
            rate_map_ptr + n * conv_out_spatial_dim,
            in_channels, input_height, input_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            height_out, width_out,
            grad_input_ptr + n * input_dim
        );

        /**
        * calculate d loss / d weight
        **/
        pad_conv2d_im2col(
            THCState_getCurrentStream(state),
            input_ptr + n * input_dim,
            rate_map_ptr + n * conv_out_spatial_dim,
            in_channels, input_height, input_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            height_out, width_out,
            col_buffer_ptr);
        
        THCudaBlas_Sgemm(state,
                't', 'n',
                M, K, N,
                1.0f,
                col_buffer_ptr, N,
                out_grad_instance_ptr, N,
                1.0f,
                grad_weight_ptr, M);
    }
//    return {grad_input, grad_weight, grad_rate_map};
    return {grad_input, grad_weight, grad_rate_map, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("forward", &pad_conv2d_forward, "perspective-aware dilated conv2d forward (CUDA)");
  m.def("backward", &pad_conv2d_backward, "perspective-aware dilated conv2d backward (CUDA)");
}
