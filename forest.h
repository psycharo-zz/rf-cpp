#ifndef FOREST_H
#define FOREST_H

#include "tree.h"

/**
 * an ensemble of trees
 */
template <typename TPatch>
class Forest
{
public:
  typedef Tree<TPatch> TTree;
  typedef std::vector<TPatch> Patches;
  
  // number of (OpenMP) threads
  static constexpr int NUM_THREADS = 20;

  Forest(int n_trees)
  {
    for (int i = 0; i < n_trees; ++i)
      m_trees.push_back(TTree());
  }

  Forest(int n_trees, int max_depth, int n_tests)
  {
    for (int i = 0; i < n_trees; ++i)
      m_trees.push_back(TTree(max_depth, n_tests));
  }

  void train(const Patches &pos, const Patches &neg, 
	     double p_pos = 0.9, double p_neg = 0.9)
  {
#pragma omp parallel for num_threads(NUM_THREADS) shared(pos,neg)
    for (size_t i = 0; i < m_trees.size(); ++i)
    {
      // making bootstrap samples
      Patches boot_pos;
      for (int idx : randi(pos.size() * p_pos, distr_param(0, pos.size()-1)))
	boot_pos.push_back(pos[idx]);

      Patches boot_neg;
      for (int idx : randi(neg.size() * p_neg, distr_param(0, neg.size()-1)))
	boot_neg.push_back(neg[idx]);
            
     m_trees[i].train(boot_pos, boot_neg);
//      m_trees[i].train_bfs(boot_pos, boot_neg);
    }
  }

  // get a single tree
  const TTree &tree(size_t i) const { return m_trees[i]; };

  // total # of trees
  inline size_t n_trees() const { return m_trees.size(); }

  vec predict(const Patches &patches) const
  {
    vec res = zeros(patches.size());
    for (auto &t : m_trees)
      res += t.predict(patches);
    return res / m_trees.size();
  }  

private:
  std::vector<TTree> m_trees;
};



#endif // FOREST_H
