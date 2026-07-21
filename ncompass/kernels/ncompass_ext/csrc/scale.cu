#include <torch/extension.h>

__global__ void scale_kernel(const float* x, float* y, float f, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * f;
}

torch::Tensor scale(torch::Tensor x, double f) {
  auto y = torch::empty_like(x);
  int n = x.numel(), threads = 256, blocks = (n + threads - 1) / threads;
  scale_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), (float)f, n);
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scale", &scale, "y = x * f"); }
