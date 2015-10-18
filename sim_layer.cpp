#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SimLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void SimLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void SimLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
void SimLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

#ifdef CPU_ONLY
STUB_GPU(SimLayer);
#endif

INSTANTIATE_CLASS(SimLayer);
REGISTER_LAYER_CLASS(Sim);

}  // namespace caffe
