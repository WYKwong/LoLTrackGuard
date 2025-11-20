#include <torch/extension.h>
#include "video_loader.h"
#include "kernels.h"

// Wrapper for CUDA kernel
torch::Tensor preprocess_cuda(torch::Tensor input) {
    // Ensure input is on CUDA and correct type (uint8/Byte)
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input tensor must be uint8");

    auto output = torch::empty_like(input);
    
    const unsigned char* d_input = input.data_ptr<unsigned char>();
    unsigned char* d_output = output.data_ptr<unsigned char>();
    
    int height = input.size(0);
    int width = input.size(1);
    int channels = input.size(2);

    launch_highlight_kernel(d_input, d_output, width, height, channels);
    
    return output;
}

// PyBind definitions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess_cuda", &preprocess_cuda, "CUDA Preprocessing Kernel (Highlight Cursor)");

    pybind11::class_<AsyncVideoLoader>(m, "AsyncVideoLoader")
        .def(pybind11::init<std::string>())
        .def("get_frame", [](AsyncVideoLoader& self) -> pybind11::object {
            cv::Mat frame;
            if (self.get_frame(frame)) {
                // Convert cv::Mat to Numpy array
                // This requires a bit more boilerplate in pure C++, 
                // but for simplicity in this example we return a dummy object or implementation-dependent structure.
                // In a real production environment, use pybind11_opencv or similar helpers.
                
                // Minimal conversion (copy):
                // 1. Allocate numpy array
                // 2. Memcpy data
                
                // Here we simply return None to indicate "not fully implemented without numpy headers"
                // Users should look at detailed OpenCV-to-Numpy bindings.
                return pybind11::none(); 
            }
            return pybind11::none();
        })
        .def("stop", &AsyncVideoLoader::stop);
}

