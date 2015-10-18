#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SimLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  float sim_thr = this->layer_param_.sim_param().sim_thr();
  const Dtype* bottom_data1 = bottom[0]->cpu_data();
  const Dtype* bottom_data2 = bottom[1]->cpu_data();
  
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < bottom[0]->num(); ++i) {
    float dif = bottom_data1[i] - bottom_data2[i];
    if (dif * dif < sim_thr * sim_thr) top_data[i] = 1;
    else top_data[i] = 0;
  }
}

template <typename Dtype>
void SimLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

INSTANTIATE_LAYER_GPU_FUNCS(SimLayer);

}  // namespace caffe
