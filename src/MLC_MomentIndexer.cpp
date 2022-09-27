#include "MLC_MomentIndexer.hpp"

// Internal includes
#include "MLC_Exceptions.hpp"

// STL includes
#include <stdexcept>
#include <string>

namespace mlc {

namespace {

unsigned int
convertNumCoefficientsToNumMoments(const unsigned int num_coefficients)
{
  const unsigned int num_max_moments = 20;
  for(unsigned int num_moments=3; num_moments<num_max_moments; ++num_moments){
    const unsigned int num_moment_coefficients = (num_moments+2) * (num_moments+1) * (num_moments) / 6 - 5;
    if(num_moment_coefficients == num_coefficients)
      return num_moments;
    MLC_ASSERT(num_moment_coefficients <= num_coefficients,
               "convertNumCoefficientsToNumMoments : Invalid number of coefficients "<<num_coefficients<<" - overshot with "<<num_moment_coefficients<<" for "<<num_moments<<" moments.");
  }
  MLC_THROW("convertNumCoefficientsToNumMoments : Invalid number of coefficients "<<num_coefficients<<".");
}

}

MomentIndexer::
MomentIndexer():
  num_coefficients_(0),
  num_moments_(0)
{

}

void
MomentIndexer::
setupForCoefficients(const unsigned int num_coefficients)
{
  num_coefficients_ = num_coefficients;
  num_moments_ = convertNumCoefficientsToNumMoments(num_coefficients_);
  indexes_.resize(num_moments_*num_moments_*num_moments_,-1);

  // THE INDEXING HERE MUST AGREE WITH THE PYTHON SCRIPTS THAT DEFINE THE INPUTS TO THE ML SCHEME
  // The indexing skips:
  // (0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,0,0)

  int idx = 0;
  for(unsigned int i=0; i<num_moments_; ++i)
    for(unsigned int j=0; j<num_moments_-i; ++j)
      for(unsigned int k=0; k<num_moments_-i-j; ++k){

        // Skip (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        if(i+j+k <= 1)
          continue;

        // Skip (2,0,0)
        if((i+j+k == 2) and (i==2))
          continue;

        // Add this index (defaults to -1 which means index doesn't exist)
        set(i,j,k,idx);
        ++idx;
      }

  MLC_ASSERT(idx == num_coefficients_,
             "MomentIndexer::setupForCoefficients : Incorrect number of coefficients found. Expected "<<num_coefficients_<<", got "<<idx<<".");

}

void
MomentIndexer::
setupForMoments(const unsigned int num_moments)
{
  num_coefficients_ = (num_moments+2)*(num_moments+1)*(num_moments)/6;
  num_moments_ = num_moments;
  indexes_.resize(num_moments_*num_moments_*num_moments_,-1);

  int idx = 0;
  for(unsigned int i=0; i<num_moments_; ++i)
    for(unsigned int j=0; j<num_moments_-i; ++j)
      for(unsigned int k=0; k<num_moments_-i-j; ++k){

        // Add this index (defaults to -1 which means index doesn't exist)
        set(i,j,k,idx);
        ++idx;
      }

  MLC_ASSERT(idx == num_coefficients_,
             "MomentIndexer::setupForMoments : Incorrect number of coefficients found. Expected "<<num_coefficients_<<", got "<<idx<<".");

}

void
MomentIndexer::
set(const unsigned int i,
    const unsigned int j,
    const unsigned int k,
    const int index)
{
  // Check that input index is within range of indexer
  MLC_ASSERT(i < num_moments_, "MLDSMCIndexer::Indexer::operator() : Index i is out of range.");
  MLC_ASSERT(j < num_moments_, "MLDSMCIndexer::Indexer::operator() : Index j is out of range.");
  MLC_ASSERT(k < num_moments_, "MLDSMCIndexer::Indexer::operator() : Index k is out of range.");

  indexes_[(i*num_moments_+j)*num_moments_+k] = index;
}

}
