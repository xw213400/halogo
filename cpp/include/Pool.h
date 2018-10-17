#ifndef __POOL_H__
#define __POOL_H__

#include <cstring>
#include <iostream>

template <class T>
class Pool
{
  public:
    Pool(const std::string& name)
    {
        _name = name;
        _pool = nullptr;
        _index = _length = 0;
    }

    void resize(std::size_t size)
    {
        if (_pool != nullptr)
        {
            T **temp = _pool;
            _pool = new T *[size];
            memcpy(_pool, temp, sizeof(T *) * _length);
            delete temp;
        }
        else
        {
            _pool = new T *[size];
        }

        for (size_t i = _length; i != size; ++i)
        {
            _pool[i] = new T();
        }

        _index = _length = size;
    }

    T *pop()
    {
        if (_index <= 0)
        {
            std::cerr << "Pool '" << _name  << "' is empty." << std::endl;
        }
        return _pool[--_index];
    }

    void push(T *t)
    {
        if (_index >= _length)
        {
            std::cerr << "Pool '" << _name  << "' can not push any more." << std::endl;
        }
        _pool[_index++] = t;
    }

    std::size_t size()
    {
        return _index;
    }

  private:
    T **_pool;
    std::size_t _length;
    std::size_t _index;
    std::string _name;
};

#endif