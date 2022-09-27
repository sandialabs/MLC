#ifndef MLC_Array_hpp_
#define MLC_Array_hpp_

// STL includes
#include <vector>

namespace mlc {

/**
 * \class TypedArray
 *
 * \brief Generic tensor for passing around datasets
 */
template<typename Scalar>
class
TypedArray
{
public:

  /**
   * \brief Base constructor
   *
   * \param[in] extent Length of array per dimension
   */
  TypedArray(const std::vector<unsigned int> & extent = {})
  {
    extent_ = extent;
    size_ = 1;
    for(unsigned int j=0; j<extent.size(); ++j)
      size_ *= extent[j];
    stride_ = std::vector<unsigned int>(extent.size()+1,1);
    for(unsigned int i=0; i<extent.size(); ++i){
      stride_[i] = 1;
      for(unsigned int j=i+1; j<extent.size(); ++j)
        stride_[i] *= extent[j];
    }

    data_ = std::vector<Scalar>(size(),Scalar(0));
  }

  /// Base access operator
  Scalar
  operator[](const unsigned int i) const
  {
    return data_[i];
  }

  /// Base access operator (non-const)
  Scalar &
  operator[](const unsigned int i)
  {
    return data_[i];
  }

  /// Access data pointer
  const Scalar *
  data() const
  {return data_.data();}

  /// Access data pointer (non-const)
  Scalar *
  data()
  {return data_.data();}

  /// Get strided access offset for a given dimension
  unsigned int
  stride(const int dim) const
  {return stride_[dim];}

  /// Get the extent vector
  const std::vector<unsigned int> &
  extent() const
  {return extent_;}

  /// Get the extent vector
  unsigned int
  extent(const int dim) const
  {return extent_[dim];}

  /// Get the total number of entries
  unsigned int
  size() const
  {return size_;}

  /// Fill array with a given value
  void
  clear(const Scalar value = Scalar(0))
  {std::fill_n(data_.data(),size(),value);}

protected:

  /// Total number of entires
  unsigned int size_;

  /// Stride per dimension
  std::vector<unsigned int> stride_;

  /// Width of array per dimension
  std::vector<unsigned int> extent_;

  /// Data storage container
  std::vector<Scalar> data_;

};

/// Default double precision array
using Array=TypedArray<double>;

/// Single precision array used with TensorFlow components
using MLArray=TypedArray<float>;

}

#endif
