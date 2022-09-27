#ifndef MLC_MomentIndexer_hpp_
#define MLC_MomentIndexer_hpp_

// STL includes
#include <vector>

#ifdef MLC_RANGE_CHECK
#include "MLC_Exceptions.hpp"
#endif

namespace mlc {

/**
 * \class MomentIndexer
 *
 * \brief Used to translate indexes for spectral coefficients and moments.
 *
 * Moments are indexed in a tetrahedral pattern:
 *
 * i+j+k<N
 *
 * where N is the total number of moments/coefficients.
 *
 * The total number of coefficients given a maximum moment index of n is
 *
 * N = (n)*(n+1)*(n+2)/6
 *
 * So:
 *
 *   n  |  N
 *   1  |  1      (0,0,0)
 *   2  |  4      (0,0,0), (1,0,0), (0,1,0), (0,0,1)
 *   3  |  10
 *   4  |  20
 *   5  |  35
 *
 * Note that for coefficients we are removing the five lowest moments, so we actually start with
 *
 * num_coeff = N - 5
 *
 */
struct
MomentIndexer
{

  /// Base constructor
  MomentIndexer();

  /**
   * \brief Setup the indexer for a given maximum moment index
   *
   * \param[in] num_moments Number of moments n (note that this not N - see class description)
   */
  void
  setupForMoments(const unsigned int num_moments);

  /**
   * \brief Setup the indexer for a the total number of coefficients (note that coefficients do not include the first 5 scalar moments)
   *
   * \param[in] num_coefficients Number of total moments N-5 (note that this not n - see class description)
   */
  void
  setupForCoefficients(const unsigned int num_coefficients);

  /**
   * \brief Get the linear coefficient/moment index given its 3-index.
   *
   * \throws If 3-index is out of range
   *
   * \param[in] i Index for dimension 0
   * \param[in] j Index for dimension 1
   * \param[in] k Index for dimension 2
   *
   * \return Linear index (-1 if linear index does not exist)
   */
  int
  operator()(const unsigned int i,
             const unsigned int j,
             const unsigned int k) const
  {
#ifdef MLC_RANGE_CHECK
    // Check that input index is within range of indexer
    MLC_ASSERT(i < num_moments_, "MLDSMCIndexer::Indexer::operator() : Index i is out of range.");
    MLC_ASSERT(j < num_moments_, "MLDSMCIndexer::Indexer::operator() : Index j is out of range.");
    MLC_ASSERT(k < num_moments_, "MLDSMCIndexer::Indexer::operator() : Index k is out of range.");
#endif

    return indexes_[(i*num_moments_+j)*num_moments_+k];
  }

  /// Get the total number of coefficients found in this indexer
  unsigned int
  getNumCoefficients() const
  {return num_coefficients_;}

  /// Get the number of moments for this indexer
  unsigned int
  getNumMoments() const
  {return num_moments_;}

protected:

  /**
   * \brief Set the linear coefficient/moment index given its 3-index.
   *
   * \throws If 3-index is out of range
   *
   * \param[in] i Index for dimension 0
   * \param[in] j Index for dimension 1
   * \param[in] k Index for dimension 2
   * \param[in] index Linear index to set
   */
  void
  set(const unsigned int i,
      const unsigned int j,
      const unsigned int k,
      const int index);

  /// Total number of coefficients
  unsigned int num_coefficients_;

  /// Number of moments per dimension
  unsigned int num_moments_;

  /// Indexer lookup table
  std::vector<int> indexes_;

};

}

#endif
