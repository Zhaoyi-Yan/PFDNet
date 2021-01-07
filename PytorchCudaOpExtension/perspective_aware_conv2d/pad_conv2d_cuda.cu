#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include "pad_conv2d.h"

#define CUDA_KERNEL_LOOP(i ,n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<(n); i+= blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N){
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__device__ float dmcn_im2col_bilinear(
    const float* bottom_data,
    const int data_width,
    const int height,
    const int width,
    float h,
    float w){

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0;
  if (h_low >=0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;

}

__device__ float dmcn_get_gradient_weight(
    float argmax_h, // offset h
    float argmax_w, // offset w
    const int h,  const int w, // coordinate
    const int height,  const int width){

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width) {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

__device__ float dmcn_get_coordinate_weight(
    float argmax_h,
    float argmax_w,
    const int height,
    const int width,
    const float* im_data,
    const int data_width,
    const int bp_dir
    ) {

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

__global__ void add_bias_kernel(
    int n,
    float* data_out,
    const float* bias,
    const int out_channels,
    const int height_out, const int width_out
){
    CUDA_KERNEL_LOOP(index, n){
        const int c_col = (index / width_out / height_out) % out_channels;
        float value = bias[c_col];
        atomicAdd(data_out + index, value);
    }
}

__global__ void calculate_dbias_kernel(
    int n,
    const float* grad_output,
    float* grad_bias,
    const int out_channels,
    const int height_out, const int width_out
){
    CUDA_KERNEL_LOOP(index, n){
        const int c_col = (index / width_out / height_out) % out_channels;
        float value = grad_output[index];
        atomicAdd(grad_bias + c_col, value);
    }
}

__global__ void pad_conv2d_im2col_kernel(
    int n,
    const float* data_im,
    const float* data_rate,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int num_channels,
    const int height_col, const int width_col,
    float* data_col
    ){
    CUDA_KERNEL_LOOP(index, n){
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int c_im = index / width_col / height_col;
        const int c_col = c_im * kernel_h * kernel_w;
        const float rate = data_rate[h_col * width_col + w_col];
        
        const int h_in = h_col * stride_h + (int)((kernel_h - 1 ) / 2);
        const int w_in = w_col * stride_w + (int)((kernel_w - 1 ) / 2);

        float* data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;
        const float* data_im_ptr = data_im + c_im * height * width;
        
        for (int i = - (int)(kernel_h / 2); i <= (int)(kernel_h / 2); ++i) {
            for (int j = - (int)(kernel_w / 2); j <= (int)(kernel_w / 2); ++j) {
                
                float val = static_cast<float>(0);
                const float h_im = h_in + i * 1 * rate;
                const float w_im = w_in + j * 1 * rate;
                if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
                    val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
                }
                *data_col_ptr = val;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

__global__ void pad_conv2d_col2im_coord_kernel(
 const int n,
 const float* data_col,
 const float* data_im,
 const float* data_rate,
 const int channels, const int height, const int width,
 const int kernel_h, const int kernel_w,
 const int stride_h, const int stride_w,
 const int height_col, const int width_col,
 float* grad_rate_map
){
   CUDA_KERNEL_LOOP(index, n){
       // the relative location in the filter
        const int j = (index / width_col / height_col) % kernel_w;
        const int i = (index / width_col / height_col / kernel_w) % kernel_h;
        const int c = index / width_col / height_col / kernel_w / kernel_h;
        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        // corrdinates of center of conv window in the image.
        const int h_in = h_out * stride_h + (int)((kernel_h - 1 ) / 2);
        const int w_in = w_out * stride_w + (int)((kernel_w - 1 ) / 2);
        const float rate = data_rate[h_out * width_col + w_out];
        
        const float cur_inv_h_data = h_in + (i - (int)((kernel_h - 1 ) / 2)) * rate;
        const float cur_inv_w_data = w_in + (j - (int)((kernel_w - 1 ) / 2)) * rate;
        
        const float reletive_i = (i - (int)((kernel_h - 1 ) / 2));
        const float reletive_j = (j - (int)((kernel_w - 1 ) / 2));
        if (reletive_i != 0 || reletive_j != 0){
            float val_h = 0;
            float val_w = 0;
            float h_weight = dmcn_get_coordinate_weight(
                cur_inv_h_data, cur_inv_w_data,
                height, width,
                data_im + c * height * width,
                width,
                0);
            float w_weight = dmcn_get_coordinate_weight(
                cur_inv_h_data, cur_inv_w_data,
                height, width,
                data_im + c * height * width,
                width,
                1);

            val_h = (h_weight) * data_col[index];
            val_w = (w_weight) * data_col[index];

            float gradient = 0;
            float tmp = val_h * reletive_i + val_w * reletive_j;
            gradient = tmp / std::sqrt(float(reletive_i * reletive_i + reletive_j * reletive_j));
            atomicAdd(grad_rate_map + h_out * width_col + w_out, gradient);
        }
   }
}

__global__ void pad_conv2d_col2im_kernel(
    const int n,
    const float* data_col,
    const float* data_rate,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* grad_im
){
    CUDA_KERNEL_LOOP(index, n){
        // the relative location in the filter
        const int j = (index / width_col / height_col) % kernel_w;
        const int i = (index / width_col / height_col / kernel_w) % kernel_h;
        const int c = index / width_col / height_col / kernel_w / kernel_h; // which channel
        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        const int h_in = h_out * stride_h + (int)((kernel_h - 1 ) / 2);
        const int w_in = w_out * stride_w + (int)((kernel_w - 1 ) / 2);
        const float rate = data_rate[h_out * width_col + w_out];
        const float cur_inv_h_data = h_in + (i - (int)((kernel_h - 1 ) / 2)) * rate;
        const float cur_inv_w_data = w_in + (j - (int)((kernel_w - 1 ) / 2)) * rate;
        const int cur_h = (int)cur_inv_h_data;
        const int cur_w = (int)cur_inv_w_data;
        const float cur_top_grad = data_col[index];
        for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 && cur_w + dx < width)
                {
                    int cur_bottom_grad_pos = (c * height + cur_h + dy) * width + cur_w + dx;
                    float weight = dmcn_get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
                    atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
                }
            }
        }
    }
}

void pad_conv2d_im2col(cudaStream_t stream,
    const float* data_im,
    const float* data_rate,
    const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
//     const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
//     const int dilation_h, const int dilation_w,
    const int height_out, const int width_out,
    float* data_col){
    int num_kernels = in_channels * height_out * width_out;
    pad_conv2d_im2col_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels,
            data_im,
            data_rate,
            height, width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            in_channels,
            height_out, width_out,
            data_col
    );
}

void pad_conv2d_col2im_coord(cudaStream_t stream,
   const float* data_col, const float* data_im, const float* data_rate,
   const int in_channels, const int height, const int width,
   const int kernel_h, const int kernel_w,
   const int stride_h, const int stride_w,
   const int height_col, const int width_col,
   float* grad_rate_map){
   int num_kernels = in_channels * kernel_h * kernel_w * height_col * width_col;
   pad_conv2d_col2im_coord_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
       num_kernels,
       data_col,
       data_im,
       data_rate,
       in_channels, height, width,
       kernel_h, kernel_w,
       stride_h, stride_w,
       height_col, width_col,
       grad_rate_map
   );
}

void pad_conv2d_col2im(cudaStream_t stream,
    const float* data_col, const float* data_rate,
    const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out,
    float* grad_im){
    int  num_kernels = in_channels * kernel_h * kernel_w * height_out * width_out;
    pad_conv2d_col2im_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        data_col,
        data_rate,
        in_channels, height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        height_out, width_out,
        grad_im
    );
}

void add_bias(cudaStream_t stream,
    float* data_out,
    const float* bias,
    const int out_channels,
    const int height_out, const int width_out
    ){
    int num_kernels = out_channels * height_out * width_out;
    add_bias_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        data_out,
        bias,
        out_channels,
        height_out, width_out
    );
}

void calculate_dbias(cudaStream_t stream,
    const float* grad_output,
    float* grad_bias,
    const int out_channels,
    const int height_out, const int width_out
    ){
    int num_kernels = out_channels * height_out * width_out;
    calculate_dbias_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        grad_output,
        grad_bias,
        out_channels,
        height_out, width_out
    );
}
