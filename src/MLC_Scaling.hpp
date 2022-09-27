#ifndef MLC_Scaling_hpp_
#define MLC_Scaling_hpp_

#define CLAMP(v,lo,hi) ((v < lo) ? lo : (hi < v) ? hi : v)

namespace mlc {

/**
 * \class Scaling
 *
 * \brief Used to transform values from one scaling (x) to another (y) - this is a linear scaling.
 */
class Scaling
{
public:

  // Default constructor - no scaling
  Scaling()
  {
    x_span_[0] = -1.;
    x_span_[1] =  1.;
    y_span_[0] = -1.;
    y_span_[1] =  1.;
  }

  /**
   * \brief Scaling constructor
   *
   * \param[in] x_min Minimum value for x
   * \param[in] x_max Maximum value for x
   * \param[in] y_min Minimum value for y
   * \param[in] y_max Maximum value for y
   */
  Scaling(const double x_min,
          const double x_max,
          const double y_min = -1.,
          const double y_max =  1.)
  {
    x_span_[0] = x_min;
    x_span_[1] = x_max;
    y_span_[0] = y_min;
    y_span_[1] = y_max;
  }

  /// Default destructor
  ~Scaling() = default;

  /// Convert from x to y
  double
  xToY(const double x) const
  {
    // Note that this clamps the variable... not sure if that is a good thing
    const double y = (y_span_[1] - y_span_[0]) * (x - x_span_[0]) / (x_span_[1] - x_span_[0]) + y_span_[0];
    return CLAMP(y,y_span_[0],y_span_[1]);
  }

  /// Convert from y to x
  double
  yToX(const double y) const
  {
    // Note that this clamps the variable... not sure if that is a good thing
    const double x = (x_span_[1] - x_span_[0]) * (y - y_span_[0]) / (y_span_[1] - y_span_[0]) + x_span_[0];
    return CLAMP(x,x_span_[0],x_span_[1]);
  }

protected:

  /// Scaling values for x
  double x_span_[2];

  /// Scaling values for y
  double y_span_[2];

};

}

#undef CLAMP

#endif
