#ifndef GAIN_H
#define GAIN_H

#include <cmath>
#include <limits>
#include <algorithm>

constexpr double ENTROPY_MIN_GAIN = std::numeric_limits<double>::lowest();

constexpr double ENTROPY_EPS = 1e-3;

inline double entropy(int n_pos, int n_neg)
{
  if (n_pos == 0 || n_neg == 0)
    return 0.0;
  double p = double(n_pos) / (n_pos + n_neg);
  // TODO: it still can be pretty small
  p = std::max(std::min(p, 1.0-ENTROPY_EPS), ENTROPY_EPS);
  return -(p*log(p) + (1.0-p)*log(1.0-p));
}

inline double entropy_gain(int n_l_pos, int n_l_neg, int n_r_pos, int n_r_neg)
{
  int n_l = n_l_pos + n_l_neg;
  int n_r = n_r_pos + n_r_neg;
  return (double(n_l) / (n_l + n_r) * entropy(n_l_pos, n_l_neg) +
	  double(n_r) / (n_l + n_r) * entropy(n_r_pos, n_r_neg));
}

constexpr double GINI_MIN_GAIN = 0.0;

inline double gini(int n_pos, int n_neg)
{
  if (n_pos + n_neg == 0)
    return 0.0;
  double p = double(n_pos) / (n_pos + n_neg);
  return 1.0 - (p*p + (1.0-p)*(1.0-p));
}

inline double gini_gain(int n_l_pos, int n_l_neg, int n_r_pos, int n_r_neg)
{
  int n_total = n_l_pos + n_l_neg + n_r_pos + n_r_neg;

  if (n_l_pos + n_l_neg == 0 || n_r_pos + n_r_neg == 0)
    return 0.0;

  return (gini(n_l_pos + n_r_pos, n_l_neg + n_r_neg) - 
	  double(n_l_pos+n_l_neg) / n_total * gini(n_l_pos, n_l_neg) -
	  double(n_r_pos+n_r_neg) / n_total * gini(n_r_pos, n_r_neg));
}

#endif // GAIN_H
