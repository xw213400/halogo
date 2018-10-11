#include <malloc.h>
#include <iostream>
#include "Pool.h"

using namespace std;

template <class T>
Pool<T>::Pool(size_t size)
{
    pool = static_cast<T **>(malloc(sizeof(nullptr) * size));
    length = size;
    index = length - 1;

    for (size_t i = 0; i != length; ++i)
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

template <class T>
size_t Pool<T>::size(void)
{
    return index;
}