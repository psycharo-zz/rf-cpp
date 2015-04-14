#ifndef TEST_H
#define TEST_H

#include <armadillo>
using namespace arma;
#include "gain.h"

// TODO: does this belong here?
inline int area(const ivec4 &a)
{
  return (a(2)-a(0)+1) * (a(3)-a(1)+1);
}

inline ivec4 random_area(int size, int min_size, int max_size)
{
  int xmin = rand() % (size - min_size + 1);
  int ymin = rand() % (size - min_size + 1);

  int w = rand() % (size - xmin);
  int h = rand() % (size - ymin);

  w = std::min(std::max(w, min_size), max_size);
  h = std::min(std::max(h, min_size), max_size);  

  return {xmin, ymin, xmin+w-1, ymin+h-1};
}

/**
 * (binary) test
 */
template <typename TPatch>
class Test
{
public:
  typedef std::vector<TPatch> Patches;
  
  static constexpr double MIN_GAIN = GINI_MIN_GAIN;

  Test() = default;
    
  // two rectangles on the patch to test
  ivec4 p, q;
  // channel
  int c = 0;
  // threshold
  double t = 0.0;
  // gain
  double gain = MIN_GAIN;

  double feature(const TPatch &patch) const
  {
    return patch.sum(p,c) / area(p) - patch.sum(q,c) / area(q);
  }

  bool operator ()(const TPatch &patch) const
  {
    return feature(patch) > t;
  }

  std::pair<Patches,Patches> apply(const Patches &patches)
  {
    Patches l, r;
    for (auto &p : patches)
      (*this)(p) ? l.push_back(p) : r.push_back(p);
    return {l, r};
  }

  static Test random(const Patches &data)
  {
    Test res;
    res.c = rand() % data.front().n_channels();
    res.p = random_area(data.front().size, 2, data.front().size);
    res.q = random_area(data.front().size, 2, data.front().size);
    res.t = res.feature(data[rand() % data.size()]);
    res.gain = MIN_GAIN;
    return res;
  }

  static Test best(const Patches &pos, const Patches &neg, int n_tests)
  {
    Test curr_best;

    if (pos.size() == 0 || neg.size() == 0)
      return curr_best;
    
    for (int i = 0; i < n_tests; ++i)
    {
      Test test = random(pos);

      int n_l_pos = 0, n_l_neg = 0, n_r_pos = 0, n_r_neg = 0;
      for (auto &p : pos)
	n_l_pos += test(p);
      n_r_pos = pos.size()-n_l_pos;
      
      for (auto &p : neg)
	n_l_neg += test(p);
      n_r_neg = neg.size()-n_l_neg;

      test.gain = gini_gain(n_l_pos, n_l_neg, n_r_pos, n_r_neg);

      if (test.gain > curr_best.gain)
	curr_best = test;
    }
    return curr_best;
  }


//   static Test best_weighted(const Patches &pos, const Patches &neg,
// 			    const vec &pos_w, const const vec &neg_w, 
// 			    int n_tests)
//   {
//     Test curr_best;

//     if (pos.size() == 0 || neg.size() == 0)
//       return curr_best;
    
//     for (int i = 0; i < n_tests; ++i)
//     {
//       Test test = random(pos);

//       double w_l_pos = 0, w_l_neg = 0, w_r_pos = 0, w_r_neg = 0;

//       for (int i = 0; i < pos.size(); ++i)
// 	test(pos[i]) ? w_l_pos += pos_w(i) : w_r_pos += pos_w(i);
//       for (int i = 0; i < neg.size(); ++i)	
// 	test(neg[i]) ? w_l_neg += neg_w(i) : w_r_neg += neg_w(i);

//       // which gain?
// //      test.gain = gini_gain(w_l_pos, w_l_neg, w_r_pos, w_r_neg);

//       if (test.gain > curr_best.gain)
// 	curr_best = test;
//     }
//     return curr_best;
//   }
  
};

#endif // TEST_H
