#ifndef MLC_CollisionModel_hpp_
#define MLC_CollisionModel_hpp_

// STL includes
#include <string>
#include <vector>
#include <memory>

// Internal includes
#include "MLC_Array.hpp"
#include "MLC_MomentIndexer.hpp"
#include "MLC_Scaling.hpp"

namespace mlc {

class TensorFlowModel;

class
CollisionModel
{
public:

  /**
   * \brief Constructor for model
   *
   * \param[in] num_points Number of points to pass into generateCoefficients call (dimension 0)
   * \param[in] model_name Name of directory containing ML model
   */
  CollisionModel(const unsigned int num_points,
                 const std::string & model_name);

  /**
   * \brief Generate the response coefficients for this collision operator
   *
   * The input moments are the conserved moments
   *
   * \f[
   * M_{ijk} = \int_{-\infty}^{\infty} v_x^i v_y^j v_z^k f dv_x dv_y dv_z
   * \f]
   *
   * The output collisional response are coefficients of the form
   *
   * \f[
   * g = g_0 \exp(- w^2) \sum_{ijk} G_{ijk} w_x^i w_y^j w_z^k
   * \f]
   *
   * where
   *
   * \f[
   * {\bf w} &= \frac{{\bf v} - {\bf u}}{\sqrt{2} \sigma}
   * \f]
   *
   * Note that the M and G are 2D arrays that are indexed using the moment indexer.
   *
   */
  void
  generateCollisionalResponse(const Array & M,
                              Array & tau,
                              Array & g0,
                              Array & ux,
                              Array & uy,
                              Array & uz,
                              Array & sigma,
                              Array & G,
                              const bool primitive_output=true) const;

  /**
   * \brief Evaluate the ML model using inputs n,T,F to get tau,G
   *
   * \note This call destroys n,T,F
   */
  void
  evaluateModel(MLArray & n,
                MLArray & T,
                MLArray & F,
                MLArray & tau,
                MLArray & G) const;

  /**
   * \brief Moment indexer used to index into M array
   */
  const MomentIndexer &
  getInputMomentIndexer() const
  {return input_moment_indexer_;}

  /**
   * \brief Moment indexer used to index into G array
   */
  const MomentIndexer &
  getOutputMomentIndexer() const
  {return output_moment_indexer_;}

  /// Get the mass for the species
  double
  getMass() const
  {return mass_;}

  /// Get the boltzmann constant
  double
  getBoltzmannConstant() const
  {return kB_;}

protected:

  void
  loadParameters(const std::string & filename);

  std::shared_ptr<TensorFlowModel> model_;

  // Indexers for dealing with front end (M, G)
  MomentIndexer input_moment_indexer_;
  MomentIndexer output_moment_indexer_;

  // Indexers for dealing with internal (ML) components
  MomentIndexer ml_input_moment_indexer_;
  MomentIndexer ml_output_moment_indexer_;

  double mass_;
  double kB_;

  // Input scaling factors
  Scaling n_scaling_;
  Scaling T_scaling_;
  std::vector<Scaling> F_scaling_;

  // Output scaling factors
  Scaling tau_scaling_;
  std::vector<Scaling> G_scaling_;

  // Scratch space for converting between various systems
  mutable Array scratch_input_moments_0_;
  mutable Array scratch_input_moments_1_;
  mutable Array scratch_output_moments_;

  mutable MLArray ml_T_dataset_;
  mutable MLArray ml_F_dataset_;
  mutable MLArray ml_G_dataset_;
  mutable MLArray ml_tau_dataset_;

};

}

#endif
