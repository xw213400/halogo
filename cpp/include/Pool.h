#ifndef __POOL_H__
#define __POOL_H__

template <class T>
class Pool
{
  public:
    Pool(size_t size);

    T *pop();

    void push(T *t);

    size_t size();

  private:
    T **pool;
    size_t length;
    size_t index;
};

#endif