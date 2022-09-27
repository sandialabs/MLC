
#include "UnitTestHarness.hpp"

#include "MLC_MomentIndexer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

ADD_TEST(MomentIndexer, Default)
{
  using namespace mlc;

  MomentIndexer mi;
  const auto & cmi = mi;

  ASSERT_EQ(mi.getNumCoefficients(), 0);
  ASSERT_EQ(mi.getNumMoments(), 0);

#ifdef MLC_RANGE_CHECK
  ASSERT_THROWS(mi(0,0,0));
  ASSERT_THROWS(cmi(0,0,0));
#endif
}

ADD_TEST(MomentIndexer, Moments)
{

  using namespace mlc;

  MomentIndexer mi;
  const auto & cmi = mi;

  const int n = 4;
  mi.setupForMoments(n);

  ASSERT_EQ(mi.getNumMoments(),static_cast<unsigned int>(n));
  ASSERT_EQ(mi.getNumCoefficients(), static_cast<unsigned int>((n+2)*(n+1)*n/6));

  for(int i=0,idx=0; i<n; ++i)
    for(int j=0; j<n-i; ++j)
      for(int k=0; k<n-i-j; ++k){
        if(i+j+k < n){
          ASSERT_EQ(mi(i,j,k), idx);
          ++idx;
        } else {
          ASSERT_EQ(mi(i,j,k), -1);
        }
      }

#ifdef MLC_RANGE_CHECK
  ASSERT_THROWS(mi(n,0,0));
  ASSERT_THROWS(mi(0,n,0));
  ASSERT_THROWS(mi(0,0,n));
  ASSERT_THROWS(cmi(n,0,0));
  ASSERT_THROWS(cmi(0,n,0));
  ASSERT_THROWS(cmi(0,0,n));
#endif

}

ADD_TEST(MomentIndexer, Coefficients)
{

  using namespace mlc;

  MomentIndexer mi;
  const auto & cmi = mi;

  const int n = 5;

  // = 1+3+6+10+15 - 5
  const int nc = 30;
  mi.setupForCoefficients(nc);

  ASSERT_EQ(mi.getNumMoments(),static_cast<unsigned int>(n));
  ASSERT_EQ(mi.getNumCoefficients(), static_cast<unsigned int>(nc));

  for(int i=0,idx=0; i<n; ++i)
    for(int j=0; j<n-i; ++j)
      for(int k=0; k<n-i-j; ++k){
        if((i+j+k < n) and (i+j+k>1) and (not (j+k==0 and i==2))){
          ASSERT_EQ(mi(i,j,k), idx);
          ++idx;
        } else {
          ASSERT_EQ(mi(i,j,k), -1);
        }
      }

#ifdef MLC_RANGE_CHECK
  ASSERT_THROWS(mi(n,0,0));
  ASSERT_THROWS(mi(0,n,0));
  ASSERT_THROWS(mi(0,0,n));
  ASSERT_THROWS(cmi(n,0,0));
  ASSERT_THROWS(cmi(0,n,0));
  ASSERT_THROWS(cmi(0,0,n));
#endif

}
