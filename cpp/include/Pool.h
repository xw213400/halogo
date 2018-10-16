#ifndef __POOL_H__
#define __POOL_H__

#include <malloc.h>
#include <iostream>

template <class T>
class Pool
{
public:
  Pool()
  {
    pool = nullptr;
    index = length = 0;
  }

  void resize(std::size_t size)
  {
    if (pool != nullptr)
    {
      T **oldPool = pool;
      pool = new T*[size];//static_cast<T **>(malloc(sizeof(nullptr) * size));
      memcpy(pool, oldPool, sizeof(T *) * length);
      delete oldPool;
    }
    else
    {
      pool = static_cast<T **>(malloc(sizeof(nullptr) * size));
    }

    for (size_t i = length; i != size; ++i)
    {
      pool[i] = new T();
    }

    index = length = size;
  }

  T *pop()
  {
    if (index <= 0)
    {
      std::cerr << "Pool is empty." << std::endl;
    }
    return pool[--index];
  }

  void push(T *t)
  {
    if (index >= length)
    {
      std::cerr << "Pool can not push any more." << std::endl;
    }
    pool[index++] = t;
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