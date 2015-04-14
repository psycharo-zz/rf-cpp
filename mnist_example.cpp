#include "tree.h"
#include "forest.h"

#include <string>
using namespace std;

#include <opencv2/opencv.hpp>

template <typename T, int NC>
Cube<T> to_arma(const cv::Mat_<cv::Vec<T, NC>> &src)
{
  vector<cv::Mat_<T>> channels;
  Cube<T> dst(src.cols, src.rows, NC);
  for (int c = 0; c < NC; ++c)
    channels.push_back({src.rows, src.cols, dst.slice(c).memptr()});
  cv::split(src, channels);	
  return dst;
}

template <typename T>
cv::Mat_<T> to_cvmat(const Mat<T> &src)
{
  return {int(src.n_cols), int(src.n_rows), const_cast<T*>(src.memptr())};
}

cube integral_image(const cube &src)
{
  cube dst(src.n_rows+1, src.n_cols+1, src.n_slices);
  for (size_t c = 0; c < src.n_slices; ++c)
    cv::integral(to_cvmat(src.slice(c)), to_cvmat(dst.slice(c)));
  return dst;
}

typedef tuple<vector<Patch>, vector<cube>, vector<cube>> MnistPool;

MnistPool read_mnist_digit(const string &path, int n_max_samples)
{
  constexpr int size = 28;

  MnistPool pool;

  vector<Patch> &patches = std::get<0>(pool);
  vector<cube> &channels = std::get<1>(pool);
  vector<cube> &ii_channels = std::get<2>(pool);

  cv::Mat src = cv::imread(path, cv::IMREAD_GRAYSCALE);
  cv::Mat_<double> srcd;
  src.convertTo(srcd, CV_64FC1, 1.0 / 255.0);
  
  int n_samples = std::min(src.rows / size, n_max_samples);

  for (int i = 0; i < n_samples; ++i)
  {
    cv::Mat_<cv::Vec<double,1>> curr = srcd.rowRange(i*size,(i+1)*size).colRange(0,size);
    channels.push_back(to_arma(curr));
    ii_channels.push_back(integral_image(channels.back()));
  }

  for (int i = 0; i < n_samples; ++i)
    patches.push_back({ivec2{0,0}, channels[i], ii_channels[i], size});

  return pool;
}


int main(int argc, char *argv[])
{
  // example on digits
  string dataset_path = "../mnist/";
  string output_path = dataset_path + "output/";
  string fmt_train = dataset_path + "train/%d.png";
  string fmt_test = dataset_path + "test/%d.png";

  constexpr int pos_digit = 0;
  constexpr int neg_digit = 2;

  int n_train = 6000;
  int n_test = 1000;

  MnistPool pos_pool = read_mnist_digit(string_format(fmt_train, pos_digit), n_train);
  auto &pos = std::get<0>(pos_pool);

  MnistPool neg_pool = read_mnist_digit(string_format(fmt_train, neg_digit), n_train);
  auto &neg = std::get<0>(neg_pool);

  MnistPool pos_pool_test = read_mnist_digit(string_format(fmt_test, pos_digit), n_test);
  auto &pos_test = std::get<0>(pos_pool_test);
  auto &pos_test_channels = std::get<1>(pos_pool_test);
  
  MnistPool neg_pool_test = read_mnist_digit(string_format(fmt_test, neg_digit), n_test);
  auto &neg_test = std::get<0>(neg_pool_test);
  auto &neg_test_channels = std::get<1>(neg_pool_test);

  vec p_test_pos, p_test_neg;
  uvec fps, fns;

  Tree<Patch> tree;

  // training a tree 
  tree.train_bfs(pos, neg);
  //tree.train(pos, neg);

  p_test_pos = tree.predict(pos_test);
  p_test_neg = tree.predict(neg_test);

  fps = find(p_test_pos < 0.5);  
  fns = find(p_test_neg > 0.5);

  cout << string_format("results for a single tree fps=%d, fns=%d", fps.size(), fns.size()) << endl;  

  constexpr int n_trees = 100;  
  Forest<Patch> forest(n_trees, 100, 10);

  // training a forest of n_trees trees (each on resampled 90% the dataset)
  forest.train(pos, neg, 0.8, 0.8);

  p_test_pos = forest.predict(pos_test);
  p_test_neg = forest.predict(neg_test);

  fps = find(p_test_pos < 0.5);  
  fns = find(p_test_neg > 0.5);

  cout << string_format("results for a forest of %d trees fps=%d, fns=%d",
  			n_trees, fps.size(), fns.size()) << endl;  
  
  return 0;
}
