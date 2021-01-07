#ifndef PAD_CONVOLUTION
#define PAD_CONVOLUTION
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
extern THCState *state;
typedef std::vector<int> TShape;

inline int ProdShape(const TShape &shape, int start, int end) {
    int res = 1;
    for(int i=start; i<end; i++) {
        res*=shape[i];
    }
    return res;
}

inline TShape SubVector(const TShape &shape, int start, int end) {
    TShape res;
    for(int i=start;i<end;i++){
        res.push_back(shape[i]);
    }
    return res;
}

void pad_conv2d_im2col(cudaStream_t stream,
    const float* data_im, const float* rate,
    const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
//     const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
//     const int dilation_h, const int dilation_w,
    const int height_out, const int width_out,
    float* data_col);

void pad_conv2d_col2im_coord(cudaStream_t stream,
    const float* data_col, const float* data_im, const float* data_rate,
    const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* grad_rate_map);

void pad_conv2d_col2im(cudaStream_t stream,
    const float* data_col, const float* data_rate,
    const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out,
    float* grad_im);

void add_bias(cudaStream_t stream,
    float* data_out,
    const float* bias,
    const int out_channels,
    const int height_out, const int width_out
    );

void calculate_dbias(cudaStream_t stream,
    const float* grad_output,
    float* grad_bias,
    const int out_channels,
    const int height_out, const int width_out
    );

#endif
