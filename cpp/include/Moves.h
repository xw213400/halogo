#ifndef __MOVES_H__
#define __MOVES_H__

#include <cstring>
#include <iostream>

template <class T>
class Moves
{
  public:
    Moves()
    {
        _array = nullptr;
        _mark = _length = 0;
    }

    void resize(std::size_t size)
    {
        if (_array != nullptr)
        {
            T **temp = _array;
            _array = new T *[size];
            memcpy(_array, temp, sizeof(T *) * _length);
            delete temp;
        }
        else
        {
            _array = new T *[size];
        }

        for (size_t i = _length; i != size; ++i)
        {
            _array[i] = new T();
        }

        _length = size;
    }

    inline T *get(std::size_t i)
    {
        if (i >= _mark)
        {
            _mark = i + 1;
        }

        return _array[i];
    }

    inline std::size_t size()
    {
        return _mark;
    }

    inline void clear()
    {
        _mark = 0;
    }

    inline std::size_t capacity()
    {
        return _length;
    }

    inline bool full()
    {
        return _mark == _length;
    }

    inline bool empty()
    {
        return _mark == 0;
    }

  private:
    T **_array;
    std::size_t _length;
    std::size_t _mark;
};

#endif