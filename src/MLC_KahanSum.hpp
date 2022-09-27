#ifndef MLC_KahanSum_hpp_
#define MLC_KahanSum_hpp_

namespace mlc
{

/**
 * \class KahanSum
 *
 * \brief Adds precision to a summation by introducing an internal value to handle a running compensation due to summation errors
 *
 * This class was defined using the source: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
 *
 */
class KahanSum
{
public:

  /// Default constructor
  KahanSum():
    sum_(0.),
    c_(0.)
  {

  }

  /// Default destructor
  ~KahanSum() = default;

  /// Get the summed value
  double
  sum() const
  {return sum_;}

  /// Add value to sum
  void
  operator +=(const double & value)
  {
    const double y = value - c_;
    const double t = sum_ + y;
    c_ = (t - sum_) - y;
    sum_ = t;
  }

protected:

  /// Sum storage
  double sum_;

  /// Kahan compensation factor
  double c_;

};

}

#endif
