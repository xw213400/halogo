#ifndef __POOL_H__
#define __POOL_H__

#include <malloc.h>
#include <iostream>

template <class T>
class Pool
{
public:
  Pool(std::size_t size)
  {
    pool = static_cast<T **>(malloc(sizeof(nullptr) * size));
    index = length = size;

    for (size_t i = 0; i != length; ++i)
    {
      pool[i] = new T();
    }
  }

  T *pop()
  {
    if (index <= 0)
    {
      std::cerr << "Pool is empty.\n";
    }
    return pool[--index];
  }

  void push(T *t)
  {
    if (index >= length)
    {
      std::cerr << "Pool can not push any more.\n";
    }
    pool[++index] = t;
  }

  std::size_t size()
  {
    return index;
  }

private:
  T **pool;
  std::size_t length;
  std::size_t index;
};

#endif