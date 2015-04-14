#ifndef WEIGHTEDTREE_H
#define WEIGHTEDTREE_H

#include <vector>
#include <stack>
#include <queue>
#include <memory>
//using namespace std;
#include <armadillo>
using namespace arma;

#include "test.h"

// TODO: this is only necessary due to an old compiler
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

#include <cstdarg>
std::string string_format(const std::string fmt, ...)
{
  int size = 512;
  std::string str;
  va_list ap;
  while (true)
  {
    str.resize(size);
    va_start(ap, fmt);
    int n = vsnprintf((char *)str.c_str(), size, fmt.c_str(), ap);
    va_end(ap);
    if (n > -1 && n < size) {
      str.resize(n);
      return str;
    }
    if (n > -1)
      size = n + 1;
    else
      size *= 2;
  }
  return str;
}

template <typename T>
T sqr(T a) { return a * a; }


class Patch
{
public:
  // TODO: add masks and raw channels?
  Patch(const ivec2 &_loc, const cube &_channels, const cube &_ii_channels, const int _size):
    loc(_loc),
    channels(_channels),
    ii_channels(_ii_channels),
    size(_size)
  {}

  inline int n_channels() const
  {
    return channels.n_slices;
  }

  inline double at(int x, int y, int c)
  {
    return channels(loc(0)+x,loc(1)+y,c);
  }
  
  // sum over given area at a given channel
  inline double sum(const ivec4 &area, int c) const
  {
    ivec4 p = {area(0)+loc(0), area(1)+loc(1), area(2)+loc(0), area(3)+loc(1)};
    return (ii_channels(p(2)+1,p(3)+1,c) + ii_channels(p(0),p(1),c) -
	    ii_channels(p(2)+1,p(1),c) - ii_channels(p(0),p(3)+1,c));
  }

  inline double sum(int c)
  {
    ivec4 p = {loc(0), loc(1), loc(0)+size-1, loc(1)+size-1};    
    return (ii_channels(p(2)+1,p(3)+1,c) + ii_channels(p(0),p(1),c) -
	    ii_channels(p(2)+1,p(1),c) - ii_channels(p(0),p(3)+1,c));
  }

  ivec2 loc;  
  const cube &channels;
  const cube &ii_channels;
  const int size;
};


template <typename TPatch>
class Tree
{
public:
  typedef std::vector<TPatch> Patches;
  typedef Test<TPatch> TTest;

  enum NodeType
  {
    SPLIT, LEAF
  };

  class Node
  {
  public:
    Node(NodeType _type):
      type{_type}
    {}
    
    NodeType type;
  };

  class Leaf : public Node
  {
  public:
    Leaf(const Patches &_pos, const Patches &_neg):
      Node{LEAF},
      pos{_pos},
      neg{_neg},
      p_pos{double(pos.size()) / (pos.size() + neg.size())}
    {}
    
    Patches pos, neg;
    double p_pos;
  };

  class Split : public Node
  {
  public:
    Split(const TTest &_test, int _left, int _right):
      Node{SPLIT},
      test{_test},
      left{_left},
      right{_right}
    {}
    
    TTest test;
    int left, right;
  };


  Tree() = default;

  Tree(int max_depth, int n_tests):
    m_max_depth(max_depth),
    m_n_tests(n_tests)
  {}

  void train_bfs(const Patches &pos_samples, const Patches &neg_samples)
  {
    typedef std::vector<std::pair<Patches, Patches>> Layer;
    
    Layer prev = {{pos_samples, neg_samples}};
    
    for (int depth = 0; depth <= m_max_depth; ++depth)
    {
      std::vector<TTest> tests(prev.size());
      Layer curr;
      if (depth != m_max_depth)
      {
        //  #pragma omp parallel for num_threads(20) 
	for (int i = 0; i < prev.size(); ++i)
	  tests[i] = TTest::best(prev[i].first, prev[i].second, m_n_tests);
      }
      int idx = m_nodes.size() + prev.size();
      for (int i = 0; i < prev.size(); ++i)
      {
	auto pos = prev[i].first;
	auto neg = prev[i].second;
	
	if (tests[i].gain == TTest::MIN_GAIN || depth == m_max_depth)
	  m_nodes.push_back(make_unique<Leaf>(pos, neg));
	else
	{
	  Patches l_pos, l_neg, r_pos, r_neg;
	  for (auto &p : pos)
	    tests[i](p) ? l_pos.push_back(p) : r_pos.push_back(p);
	  for (auto &p : neg)
	    tests[i](p) ? l_neg.push_back(p) : r_neg.push_back(p);

	  m_nodes.push_back(make_unique<Split>(tests[i], idx, idx+1));
	  idx += 2;
	  
	  curr.emplace_back(l_pos, l_neg);
	  curr.emplace_back(r_pos, r_neg);
	}
      }
      std::swap(prev, curr);
    }
  }
    
  void train(const Patches &pos_samples, const Patches &neg_samples)
  {
    // for breadth-first it should be a queue
    std::stack<std::tuple<Patches, Patches, int, int*>> toprocess;

    // root doesn't have a parent
    int dummy;    
    toprocess.emplace(pos_samples, neg_samples, 0, &dummy);

    while (!toprocess.empty())
    {
      auto &pos = std::get<0>(toprocess.top());
      auto &neg = std::get<1>(toprocess.top());
      int depth = std::get<2>(toprocess.top());
      int *parent_idx = std::get<3>(toprocess.top());

      // it is a split node
      auto test = TTest::best(pos, neg, m_n_tests);
      // stopping criteria
      if (test.gain == TTest::MIN_GAIN || depth == m_max_depth)
      {
	*parent_idx = m_nodes.size();
	m_nodes.push_back(make_unique<Leaf>(pos, neg));
	toprocess.pop();
	continue;
      }

      Patches l_pos, l_neg, r_pos, r_neg;
      for (auto &p : pos)
	test(p) ? l_pos.push_back(p) : r_pos.push_back(p);
      for (auto &p : neg)
	test(p) ? l_neg.push_back(p) : r_neg.push_back(p);

      *parent_idx = m_nodes.size();

      m_nodes.push_back(make_unique<Split>(test, 0, 0));
      auto *split = static_cast<Split*>(m_nodes.back().get());

      toprocess.pop();
      toprocess.emplace(l_pos, l_neg, depth+1, &(split->left));
      toprocess.emplace(r_pos, r_neg, depth+1, &(split->right));      
    }
  }

  const Leaf *locate(const TPatch &patch) const
  {
    auto *curr = m_nodes.front().get();
    while (curr->type != LEAF)
    {
      auto *s = static_cast<Split*>(curr);
      curr = m_nodes[s->test(patch) ? s->left : s->right].get();
    }
    return static_cast<Leaf*>(curr);
  }

  // predicts p(c=1|I)
  vec predict(const Patches &patches) const
  {
    vec p_pos = zeros(patches.size(),1);
    for (size_t i = 0; i < patches.size(); ++i)
      p_pos(i) = locate(patches[i])->p_pos;
    return p_pos;
  }

  void print(std::ostream &os)
  {
    std::stack<std::pair<Node*,int>> tovisit;

    tovisit.push(make_pair(m_nodes.front().get(), 1));
    while (!tovisit.empty())
    {
      Node *curr;
      int depth;
      tie(curr, depth) = tovisit.top();
      tovisit.pop();

      if (curr->type == SPLIT)
      {
	auto *split = static_cast<Split*>(curr);
	os << std::string(depth, '-')
	   << string_format("Split(GAIN:%2.2f)", split->test.gain) << endl;
	tovisit.push({m_nodes[split->left].get(), depth+1});
	tovisit.push({m_nodes[split->right].get(), depth+1});      
      }
      else
      {
	auto *leaf = static_cast<Leaf*>(curr);
	os << std::string(depth, '-')
	   << string_format("Leaf(%d,%d)", leaf->pos.size(), leaf->neg.size()) << endl;
      }
    }
  }

private:
  int m_max_depth = 10;
  int m_n_tests = 1000;

  std::vector<std::unique_ptr<Node>> m_nodes;
};

#endif // WEIGHTEDTREE_H

