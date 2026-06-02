#include <torch/extension.h>

// 声明 CUDA Kernel 启动函数 (在 splat_cuda_kernel.cu 中实现)
torch::Tensor splat_forward(
    const torch::Tensor u_sorted, 
    const torch::Tensor v_sorted, 
    const torch::Tensor colors_sorted, 
    int H, 
    int W, 
    int point_size);

// 定义 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splat_forward", &splat_forward, "Point Splatting (CUDA)");
}