#ifndef __POOL_H__
#define __POOL_H__

template <class T>
class Pool
{
  public:
    Pool(int size);

    T *pop();

    void push(T *t);

  private:
    T **pool;
    int length;
    int index;
};

#endif