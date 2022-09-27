
#include "UnitTestHarness.hpp"

#include "MLC_Scaling.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

ADD_TEST(Scaling, Basic)
{

  using namespace mlc;

  // Default scaling
  {
    Scaling a;

    ASSERT_NEARLY_EQ(a.xToY(0.5),   0.5, 1.e-14);
    ASSERT_NEARLY_EQ(a.xToY(-0.5), -0.5, 1.e-14);

    ASSERT_NEARLY_EQ(a.yToX(0.5),   0.5, 1.e-14);
    ASSERT_NEARLY_EQ(a.yToX(-0.5), -0.5, 1.e-14);

  }

  // Scaling around x
  {
    Scaling a(-2.,0.);

    ASSERT_NEARLY_EQ(a.xToY(-0.5),  0.5, 1.e-14);
    ASSERT_NEARLY_EQ(a.xToY(-1.5), -0.5, 1.e-14);

    ASSERT_NEARLY_EQ(a.yToX(0.5),  -0.5, 1.e-14);
    ASSERT_NEARLY_EQ(a.yToX(-0.5), -1.5, 1.e-14);

  }

  // Scaling around x and y
  {
    Scaling a(-2.,0.,0.,2.);

    ASSERT_NEARLY_EQ(a.xToY(-0.5),  1.5, 1.e-14);
    ASSERT_NEARLY_EQ(a.xToY(-1.5),  0.5, 1.e-14);

    ASSERT_NEARLY_EQ(a.yToX(0.5),  -1.5, 1.e-14);
    ASSERT_NEARLY_EQ(a.yToX(1.5), -0.5, 1.e-14);

  }

}
