/*
Copyright 2020 - 2021 MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include <torch/extension.h>
#include "utils/common_utils.h"

torch::Tensor BilateralFilterCpu(torch::Tensor input,  torch::Tensor output, float spatial_sigma, float color_sigma);
torch::Tensor BilateralFilterPHLCpu(torch::Tensor input, float spatial_sigma, float color_sigma);

#ifdef WITH_CUDA
torch::Tensor BilateralFilterCuda(torch::Tensor input, torch::Tensor output, float spatial_sigma, float color_sigma);
torch::Tensor BilateralFilterPHLCuda(torch::Tensor input, float spatial_sigma, float color_sigma);
#endif

torch::Tensor BilateralFilter(torch::Tensor input, torch::Tensor output, float spatial_sigma, float color_sigma, bool usePHL) {
  torch::Tensor (*filterFunction)(torch::Tensor, torch::Tensor, float, float);

#ifdef WITH_CUDA
  if (torch::cuda::is_available() && input.is_cuda()) {
    CHECK_CONTIGUOUS_CUDA(input);
    filterFunction =  &BilateralFilterCuda;
  } else {
    filterFunction =  &BilateralFilterCpu;
  }
#else
  filterFunction =  &BilateralFilterCpu;
#endif

  return filterFunction(input, output, spatial_sigma, color_sigma);
}

torch::Tensor BilateralFilterPHL(torch::Tensor input, float spatial_sigma, float color_sigma, bool usePHL) {
  torch::Tensor (*filterFunction)(torch::Tensor, float, float);

#ifdef WITH_CUDA
  if (torch::cuda::is_available() && input.is_cuda()) {
    CHECK_CONTIGUOUS_CUDA(input);
    filterFunction =  &BilateralFilterPHLCuda;
  } else {
    filterFunction =  &BilateralFilterPHLCpu;
  }
#else
  filterFunction =  &BilateralFilterPHLCpu;
#endif

  return filterFunction(input, spatial_sigma, color_sigma);
}

torch::Tensor JointBilateralFilterCuda(torch::Tensor input, torch::Tensor input2, torch::Tensor output, float spatial_sigma, float color_sigma);
torch::Tensor JointBilateralFilter(torch::Tensor input, torch::Tensor input2, torch::Tensor output, float spatial_sigma, float color_sigma, bool usePHL) {
  
  // torch::Tensor (*filterFunction)(torch::Tensor, float, float);
  // filterFunction = &JointBilateralFilterCuda;

  return JointBilateralFilterCuda(input, input2, output, spatial_sigma, color_sigma);
}
