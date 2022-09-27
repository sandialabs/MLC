#ifndef MLC_Matrix_hpp_
#define MLC_Matrix_hpp_

// STL includes
#include <ostream>

namespace mlc {

/**
 * \class TVector
 *
 * \brief Templated vector class - no idea why this is here - we generally use std::vector.
 *
 * This is used in the TensorFlow graph to store things like biases.
 */
template<typename T>
class TVector
{
public:

  TVector():
    extent_(0),
    data_(nullptr)
  {

  }

  TVector(const int length,
          const T fill = 0):
    extent_(length),
    data_(new T[length])
  {
    for(int i=0; i<length; ++i) data_[i] = fill;
  }

  ~TVector()
  {
    if(data_ != nullptr) delete [] data_;
    data_ = nullptr;
  }

  int
  size() const
  {return extent_;}

  T &
  operator[](const int i)
  {return data_[i];}

  T
  operator[](const int i) const
  {return data_[i];}

  T &
  operator()(const int i)
  {return data_[i];}

  T
  operator()(const int i) const
  {return data_[i];}

  T *
  data()
  {return data_;}

  const T *
  data() const
  {return data_;}

protected:
  int extent_;
  T *data_;
};


/**
 * \class TMAtrix
 *
 * \brief Templated matrix class.
 *
 * This is used in the TensorFlow graph to store things like weights tensors.
 */
template<typename T>
class TMatrix
{
public:

  TMatrix():
    extent_0_(0),
    extent_1_(0),
    data_(nullptr)
  {

  }

  TMatrix(const int N,
          const int M,
          const T fill = 0.):
    extent_0_(N),
    extent_1_(M),
    data_(new T[N*M])
  {
    for(int i=0; i<N*M; ++i) data_[i] = fill;
  }

  ~TMatrix()
  {
    if(data_ != nullptr) delete [] data_;
    data_ = nullptr;
  }

  int
  width() const
  { return extent_1_; }

  int
  height() const
  { return extent_0_; }

  int
  size() const
  {return extent_0_ * extent_1_;}

  T *
  operator[](unsigned int i)
  { return data_+i*extent_1_; }

  const T *
  operator[](unsigned int i) const
  { return data_+i*extent_1_; }

  T &
  operator()(unsigned int i, unsigned int j)
  { return data_[i*extent_1_+j]; }

  T
  operator()(unsigned int i, unsigned int j) const
  { return data_[i*extent_1_+j]; }

  T *
  data()
  {return data_;}

  const T *
  data() const
  {return data_;}

protected:
  int extent_0_, extent_1_;
  T * data_;
};

template<typename T>
void
dot(const TMatrix<T> & A,
    const TMatrix<T> & B,
    TMatrix<T> & C)
{
  const int H = A.height();
  const int W = A.width();
  const int L = B.width();

  for(int i=0; i<H; ++i){
    for(int j=0; j<L; ++j){
      T & value = C(i,j) = 0;
      for(int k=0; k<W; ++k)
        value += A(i,k) * B(k,j);
    }
  }
}

template<typename T>
void
dot(const TMatrix<T> & A,
    const TVector<T> & B,
    TVector<T> & C)
{
  const int imax = C.size();
  const int jmax = B.size();

  for(int i=0; i<imax; ++i){
    T & value = C(i) = 0;
    for(int j=0; j<jmax; ++j)
      value += A(i,j) * B(j);
  }
}

template<typename T>
void
dot(const TVector<T> & A,
    const TMatrix<T> & B,
    TVector<T> & C)
{
  const int imax = A.size();
  const int jmax = C.size();

  for(int j=0; j<jmax; ++j){
    T & value = C(j) = 0;
    for(int i=0; i<imax; ++i)
      value += A(i) * B(i,j);
  }
}

using Matrix=TMatrix<double>;
using Vector=TVector<double>;

}

template<typename T>
std::ostream &
operator<<(std::ostream & os,
           const mlc::TVector<T> & vector)
{
  for(int i=0; i<vector.size(); ++i){
    os << vector(i);
    if(i < vector.size()-1)
      os << " ";
  }
  return os;
}

template<typename T>
std::ostream &
operator<<(std::ostream & os,
           const mlc::TMatrix<T> & matrix)
{
  for(int i=0; i<matrix.height(); ++i){
    for(int j=0; j<matrix.width(); ++j){
      os << matrix(i,j);
      if(j < matrix.width()-1)
        os << " ";
    }
    os << "\n";
  }
  return os;
}

#endif
