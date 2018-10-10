#include "pool.h"

template <class T>
Pool<T>::Pool(int size)
{
    pool = static_cast<T **>(malloc(sizeof(nullptr) * size));
    length = size;
    index = length - 1;

    for (int i = 0; i != length; ++i)
    {
        pool[i] = new T();
    }
}

template <class T>
T *Pool<T>::pop()
{
    return pool[index--];
}

template <class T>
void Pool<T>::push(T *t)
{
    pool[++index] = t;
    if (index > length)
    {
        cerr << "Pool can not push any more.\n";
    }
}