#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
// 引入这个头文件来定义 C10_CUDA_CHECK 宏
#include <c10/cuda/CUDAException.h> 

// CUDA Kernel: 执行深度排序后的点云 Splatting (Z-Buffering)
__global__ void splatting_kernel(
    const float* __restrict__ u_sorted,      // 形状 [N]
    const float* __restrict__ v_sorted,      // 形状 [N]
    const float* __restrict__ colors_sorted, // 形状 [N, 3]
    float* __restrict__ rendered_image,      // 形状 [H, W, 3]
    float* __restrict__ render_mask,         // 形状 [H, W]
    const int N,                             // 有效点的数量
    const int H,
    const int W,
    const int point_size)
{
    // 每个线程处理一个点，后面处理的线程（近点）覆盖前面处理的线程（远点）。
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        // 获取当前点的像素坐标（已四舍五入）和颜色
        // 注意：这里使用 roundf() 将浮点坐标转换为最近的整数索引
        int u = (int)roundf(u_sorted[i]);
        int v = (int)roundf(v_sorted[i]);
        float r = colors_sorted[i * 3 + 0];
        float g = colors_sorted[i * 3 + 1];
        float b = colors_sorted[i * 3 + 2];

        int half_size = point_size / 2;
        
        // 计算方形区域的边界，并钳制到图像边界
        int u_min = max(0, u - half_size);
        int u_max = min(W, u + half_size + 1);
        int v_min = max(0, v - half_size);
        int v_max = min(H, v + half_size + 1);
        
        // 在方形区域内进行 Splatting (利用排序实现 Z-Buffering)
        for (int row = v_min; row < v_max; ++row) {
            for (int col = u_min; col < u_max; ++col) {
                // 写入颜色 
                rendered_image[(row * W + col) * 3 + 0] = r;
                rendered_image[(row * W + col) * 3 + 1] = g;
                rendered_image[(row * W + col) * 3 + 2] = b;
                
                // 写入蒙版
                render_mask[row * W + col] = 1.0f;
            }
        }
    }
}

// C++ 接口函数，用于启动 Kernel
torch::Tensor splat_forward(
    const torch::Tensor u_sorted, 
    const torch::Tensor v_sorted, 
    const torch::Tensor colors_sorted, 
    int H, 
    int W, 
    int point_size) 
{
    // 检查输入是否是 CUDA 张量且是 float 类型
    TORCH_CHECK(u_sorted.is_cuda() && v_sorted.is_cuda() && colors_sorted.is_cuda(), 
                "Inputs must be CUDA tensors.");
    TORCH_CHECK(u_sorted.dtype() == torch::kFloat && colors_sorted.dtype() == torch::kFloat, 
                "Inputs must be float tensors.");

    int N = u_sorted.size(0);
    if (N == 0) {
        // 如果没有点，返回空图像和空蒙版
        return torch::cat({
            torch::zeros({H, W, 3}, colors_sorted.options()).contiguous(),
            torch::zeros({H, W, 1}, colors_sorted.options()).contiguous()
        }, 2);
    }
    
    // 准备输出张量 (在 GPU 上)
    auto rendered_image = torch::zeros({H, W, 3}, colors_sorted.options()).contiguous();
    auto render_mask = torch::zeros({H, W}, colors_sorted.options()).contiguous();

    // 设置 CUDA 启动配置
    const int THREADS = 512;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // 启动 Kernel
    splatting_kernel<<<BLOCKS, THREADS>>>(
        u_sorted.data_ptr<float>(), 
        v_sorted.data_ptr<float>(), 
        colors_sorted.data_ptr<float>(), 
        rendered_image.data_ptr<float>(), 
        render_mask.data_ptr<float>(),
        N, H, W, point_size
    );
    
    // 确保 Kernel 执行完毕
    C10_CUDA_CHECK(cudaGetLastError());

    // 将蒙版维度扩展以便连接 [H, W] -> [H, W, 1]
    auto mask_expanded = render_mask.unsqueeze(-1).contiguous();
    
    // 连接渲染图像和蒙版 [H, W, 4]
    return torch::cat({rendered_image, mask_expanded}, 2);
}