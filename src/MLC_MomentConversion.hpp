#ifndef MLC_MomentConversion_hpp_
#define MLC_MomentConversion_hpp_

// STL includes
#include <vector>
#include <cmath>
#include <stdexcept>

// Internal includes
#include "MLC_MomentIndexer.hpp"
#include "MLC_Matrix.hpp"
#include "MLC_KahanSum.hpp"

namespace mlc {

/**
 * \brief Get the transform matrix that converts between moments and Hermite polynomials.
 *
 * \param[in] N Get the matrix with a width up to this size (maximum 20)
 */
Matrix
buildTransformMatrix(const unsigned int N);

/**
 * \brief Get the inverse of the transform matrix that converts between moments and Hermite polynomials.
 *
 * \param[in] N Get the matrix with a width up to this size (maximum 20)
 */
Matrix
buildInverseTransformMatrix(const unsigned int N);

/**
 * \brief Get the binomial coefficients up to a given number
 *
 * \param[in] N Number of binomial coefficients to include
 */
Matrix
buildBinomialCoefficientMatrix(const unsigned int N);

/**
 * The following calls are used to move back and forth between our 4 representations:
 *
 * Two moment forms:
 *
 * Conserved Moments: \f$ \Gamma_{ijk} = m int_{-\infty}^{\infty} v_x^i v_y^j v_z^k f(v) d^3v \f$
 * Normalized Primitive Moments: \f$ \Lambda_{ijk} = int_{-\infty}^{\infty} w_x^i w_y^j w_z^k f(w) d^3w \f$
 *
 * Two spectral forms:
 *
 * Hermite Coefficients (F_{ijk}): \f$ f = f_0 \exp(-w^2) \sum_{ijk} F_{ijk} H_i(w_x) H_j(w_y) H_k(w_z) \f$
 * Primitive Coefficients (P_{ijk}): \f$ f = p_0 \exp(-w^2) \sum_{ijk} P_{ijk} w_x^i w_y^j w_z^k \f$
 *
 * with
 *
 * \f[
 * f_0 = \frac{n}{(\sqrt{2 \sqrt{\pi}} \sigma)^3) \\
 * p_0 = \frac{n}{(\sqrt{2 \pi} \sigma)^3).
 * \f]
 *
 * where \f$ n \f$ is the number density and \f$ \sigma \f$ is the thermal velocity.
 *
 * We use the primitive coefficients to evaluate the spectral form on a mesh (Hermite polynomials are slow to generate).
 *
 */

/**
 * \brief Convert from the conserved moments to the normalized primitive moments using a binomial factorization with some special scaling.
 *
 * \param[in] mass Particle mass
 * \param[in] n Number density
 * \param[in] ux Flow velocity in x direction
 * \param[in] uy Flow velocity in y direction
 * \param[in] uz Flow velocity in z direction
 * \param[in] P Pressure
 * \param[in] moment_indexer Moment indexer for the given moment set
 * \param[in] conserved_moments Input conserved moments indexed using moment_indexer
 * \param[out] normalized_primitive_moments Output primitive moments indexed using moment_indexer
 */
inline
void
convertConservedMomentsToNormalizedPrimitiveMoments(const double mass,
                                                    const double n,
                                                    const double ux,
                                                    const double uy,
                                                    const double uz,
                                                    const double P,
                                                    const MomentIndexer & moment_indexer,
                                                    const double * const __restrict__ conserved_moments,
                                                    double * const __restrict__ normalized_primitive_moments)
{

  /**
   * Normalized primitive moments are a scaled version of the primitive moments.
   *
   * The normalized primitive moments are defined as
   *
   * \f[
   * \Lambda_{ijk} = int_{-\infty}^{\infty} w_x^i w_y^j w_z^k f(w) d^3w.
   * \f]
   *
   * The important part is what the system is integrated over (\f$ d^3w \f$ vs \f$ d^3v \f$ - i.e. they are scaled by \f$ m (\sqrt{2} \sigma)^3 \f$ vs the actual normalized primitive moments)
   *
   * We are going to generate the normalized primitive moments using the conserved moments, defined
   *
   * \f[
   * \Gamma_{ijk} = m int_{-\infty}^{\infty} v_x^i v_y^j v_z^k f(v) d^3v.
   * \f]
   *
   * This is accomplished using the definition \f$ w = (v - u) / (\sqrt{2} \sigma) \f$, which leads to a binomial factorization of the form
   *
   * \f[
   * \Lambda_{ijk} = 1 / (\sqrt{2} \sigma)^{a+b+c+3} / m \sum_{a=0}^{i} \sum_{b=0}^{j} \sum_{c=0}^{k} (i a) (j b) (k c) (-u_x)^{i-a} (-u_y)^{j-b} (-u_z)^{k-c} \Gamma_{abc}
   * \f]
   *
   */

  // We define a quantity (\sqrt{2} \sigma)
  const double scaled_sigma = std::sqrt(2. * P / n / mass);

  const unsigned int num_moments = moment_indexer.getNumMoments();
  if(num_moments >= 20) throw std::logic_error("convertConservedMomentsToNormalizedPrimitiveMoments : Too many moments.");
  static const Matrix bc =  buildBinomialCoefficientMatrix(20);

  // This is three variable binomial expansion
  for(unsigned int i=0, pidx=0; i<num_moments; ++i){
    for(unsigned int j=0; j<num_moments-i; ++j){
      for(unsigned int k=0; k<num_moments-i-j; ++k, ++pidx){

        // This kind of algorithm is plagued with a wide variety of large and small numbers so we increase the precision of the summation
        KahanSum value;

        // Note that we have an inverse mass here because the conserved moments are assumed to by scaled by the mass
        const double mult_ijk = 1. / std::pow(scaled_sigma,i+j+k+3) / mass;
        for(unsigned int a=0; a<=i; ++a){
          const double mult_ijk_a = mult_ijk * bc(i,a) * std::pow(-ux,i-a);
          for(unsigned int b=0; b<=j; ++b){
            const double mult_ijk_ab = mult_ijk_a * bc(j,b) * std::pow(-uy,j-b);
            for(unsigned int c=0; c<=k; ++c){
              const double mult_ijk_abc = mult_ijk_ab * bc(k,c) * std::pow(-uz,k-c);
              value += mult_ijk_abc * conserved_moments[moment_indexer(a,b,c)];
            }
          }
        }
        normalized_primitive_moments[pidx] = value.sum();
      }
    }
  }

}

/**
 * \brief Convert from the normalized primitive moments to the conserved moments using a binomial expansion.
 *
 * \param[in] mass Particle mass
 * \param[in] n Number density
 * \param[in] ux Flow velocity in x direction
 * \param[in] uy Flow velocity in y direction
 * \param[in] uz Flow velocity in z direction
 * \param[in] P Pressure
 * \param[in] moment_indexer Moment indexer for the given moment set
 * \param[in] normalized_primitive_moments Input primitive moments indexed using moment_indexer
 * \param[out] conserved_moments Output conserved moments indexed using moment_indexer
 */
inline
void
convertNormalizedPrimitiveMomentsToConservedMoments(const double mass,
                                                    const double n,
                                                    const double ux,
                                                    const double uy,
                                                    const double uz,
                                                    const double P,
                                                    const MomentIndexer & moment_indexer,
                                                    const double * const __restrict__ normalized_primitive_moments,
                                                    double * const __restrict__ conserved_moments)
{

  /**
   * The conserved moments are defined
   *
   * \f[
   * \Gamma_{ijk} = m int_{-\infty}^{\infty} v_x^i v_y^j v_z^k f(v) d^3v.
   * \f]
   *
   * The normalized primitive moments are defined as
   *
   * \f[
   * \Lambda_{ijk} = int_{-\infty}^{\infty} w_x^i w_y^j w_z^k f(w) d^3w.
   * \f]
   *
   * Much like the `convertNormalizedPrimitiveMomentsToConservedMoments` call, we convert one to the other using a binomial expansion using
   * the definition \f$ w = (v - u) / (\sqrt{2} \sigma) \f$.
   *
   * \f[
   * \Gamma_{ijk}
   * = m (\sqrt{2} \sigma)^{3} int_{-\infty}^{\infty} (\sqrt{2} \sigma w_x + u_x)^i (\sqrt{2} \sigma w_y + u_y)^j (\sqrt{2} \sigma w_z + u_z^k f(w) d^3w \\
   * = m \sum\limits_{a,b,c=0}^{a<=i,b<=j,c<=k} (\sqrt{2} \sigma)^{3+a+b+c} u_x^{i-a} u_y^{j-b} u_z^{k-c} int_{-\infty}^{\infty} w_x^a w_y^b w_z^c f(w) d^3w \\
   * = m \sum\limits_{a,b,c=0}^{a<=i,b<=j,c<=k} (\sqrt{2} \sigma)^{3+a+b+c} u_x^{i-a} u_y^{j-b} u_z^{k-c} \Lambda_{abc}
   * \f]
   *
   */

  // We define a quantity (\sqrt{2} \sigma)
  const double scaled_sigma = std::sqrt(2. * P / n / mass);

  const unsigned int num_moments = moment_indexer.getNumMoments();
  if(num_moments >= 20) throw std::logic_error("convertNormalizedPrimitiveMomentsToConservedMoments : Too many moments.");
  static const Matrix bc =  buildBinomialCoefficientMatrix(20);

  // This is three variable binomial expansion
  for(unsigned int i=0, cidx=0; i<num_moments; ++i){
    for(unsigned int j=0; j<num_moments-i; ++j){
      for(unsigned int k=0; k<num_moments-i-j; ++k, ++cidx){

        // This kind of algorithm is plagued with a wide variety of large and small numbers so we increase the precision of the summation
        KahanSum value;

        // Note that we have an inverse mass here because the conserved moments are assumed to by scaled by the mass
        const double mult_ijk = std::pow(scaled_sigma,3) * mass;
        for(unsigned int a=0; a<=i; ++a){
          const double mult_ijk_a = mult_ijk * bc(i,a) * std::pow(ux,i-a) * std::pow(scaled_sigma,a);
          for(unsigned int b=0; b<=j; ++b){
            const double mult_ijk_ab = mult_ijk_a * bc(j,b) * std::pow(uy,j-b) * std::pow(scaled_sigma,b);
            for(unsigned int c=0; c<=k; ++c){
              const double mult_ijk_abc = mult_ijk_ab * bc(k,c) * std::pow(uz,k-c) * std::pow(scaled_sigma,c);
              value += mult_ijk_abc * normalized_primitive_moments[moment_indexer(a,b,c)];
            }
          }
        }
        conserved_moments[cidx] = value.sum();
      }
    }
  }

}

/**
 * \brief Convert from the normalized primitive moments to the Hermite coefficients using the transform matrices.
 *
 * \param[in] mass Particle mass
 * \param[in] n Number density
 * \param[in] P Pressure
 * \param[in] moment_indexer Moment indexer for the given moment set
 * \param[in] normalized_primitive_moments Input primitive moments indexed using moment_indexer
 * \param[out] hermite_coefficients Output Hermite coefficients indexed using moment_indexer
 */
inline
void
convertNormalizedPrimitiveMomentsToHermiteCoefficients(const double mass,
                                                       const double n,
                                                       const double P,
                                                       const MomentIndexer & moment_indexer,
                                                       const double * const __restrict__ normalized_primitive_moments,
                                                       double * const __restrict__ hermite_coefficients)
{

  /**
   *
   * Hermite coefficients define a distribution of the form
   *
   * \f[
   * f = f_0 \exp(-w^2) \sum_{ijk} F_{ijk} H_i(w_x) H_j(w_y) H_k(w_z)
   * \f]
   *
   * with scaled random velocity
   *
   * \f[
   * w = \frac{v - u}{\sqrt{2} \sigma},
   * \f]
   *
   * thermal velocity
   *
   * \f[
   * \sigma = \sqrt{P / rho},
   * \f]
   *
   * and scaling coefficient
   *
   * \f[
   * f_0 = \frac{n}{(\sqrt{2 \sqrt{\pi}} \sigma)^3}.
   * \f]
   *
   * Our goal is to define the Hermite coefficients \f$ F_{ijk} \f$ given the normalized primitive moments
   *
   * \f[
   * \Lambda_{ijk} = int_{-\infty}^{\infty} w_x^i w_y^j w_z^k f d^3w.
   * \f]
   *
   * If we integrate the random scaled moments of \f$ f \f$ we can get a direct comparison
   *
   * \f[
   * \Lambda_{ijk}
   * =
   * \int_{-\infty}^{\infty} w_x^i w_y^j w_z^k f \, d^3w
   * =
   * f_0 \int_{-\infty}^{\infty} w_x^i w_y^j w_z^k \exp(-w^2) \sum_{lmn} F_{lmn} H_l(w_x) H_m(w_y) H_n(w_z) \, d^3w
   * =
   * f_0 \sum_{lmn} M_{il} M_{jm} M_{kn} F_{lmn}
   * \f]
   *
   * where we have defined the "Transform Matrix"
   *
   * \f[
   * M_{ij} = \int_{-\infty}^{\infty} x^i H_j(x) \exp(-x^2) \, dx.
   * \f]
   *
   * We define the inverse transformation matrix \f$ M^{-1}_{ij} \f$ such that
   *
   * \f[
   * \sum_j M^{-1}_{ij} M_{jk} = \sum_j M_{ij} M^{-1}_{jk} = \delta_{ik}
   * \f]
   *
   * which is used to get the Hermite coefficients from the normalized primitive moments
   *
   * \f[
   * F_{ijk} = \frac{1}{f_0} \sum_{lmn} M^{-1}_{il} M^{-1}_{jm} M^{-1}_{kn} \Lambda_{lmn}.
   * \f]
   *
   */

  const unsigned int num_moments = moment_indexer.getNumMoments();
  if(num_moments >= 20) throw std::logic_error("convertNormalizedPrimitiveToHermite : Transform matrix is too small.");
  static const Matrix inverse_transform_matrix = buildInverseTransformMatrix(20);

  const double sigma = std::sqrt(P / n / mass);
  const double f0 = n / std::pow(std::sqrt(2. * std::sqrt(M_PI)) * sigma, 3);

  for(unsigned int i=0, hidx=0; i<num_moments; ++i){
    for(unsigned int j=0; j<num_moments-i; ++j){
      for(unsigned int k=0; k<num_moments-i-j; ++k, ++hidx){

        // We are summing over a lot of things so we use a larger double
        KahanSum value;

        // The transformation matrix is lower triangular
        for(unsigned int l=0; l<=i; ++l){
          const double mult_l = inverse_transform_matrix(i,l);
          for(unsigned int m=0; m<=j; ++m){
            const double mult_lm = mult_l * inverse_transform_matrix(j,m);
            for(unsigned int n=0; n<=k; ++n){
              const int pidx = moment_indexer(l,m,n);
              const double mult_lmn = mult_lm * inverse_transform_matrix(k,n);
              value += mult_lmn * normalized_primitive_moments[pidx];
            }
          }
        }

        hermite_coefficients[hidx] = value.sum() / f0;

      }
    }
  }

}


/**
 * \brief Convert from the normalized primitive moments to the Hermite coefficients using the transform matrices.
 *
 * \param[in] mass Particle mass
 * \param[in] n Number density
 * \param[in] P Pressure
 * \param[in] moment_indexer Moment indexer for the given moment set
 * \param[in] hermite_coefficients Input Hermite coefficients indexed using moment_indexer
 * \param[out] normalized_primitive_moments Output primitive moments indexed using moment_indexer
 */
inline
void
convertHermiteCoefficientsToNormalizedPrimitiveMoments(const double mass,
                                                       const double n,
                                                       const double P,
                                                       const MomentIndexer & moment_indexer,
                                                       const double * const __restrict__ hermite_coefficients,
                                                       double * const __restrict__ normalized_primitive_moments)
{

  /**
   *
   * Hermite coefficients define a distribution of the form
   *
   * \f[
   * f = f_0 \exp(-w^2) \sum_{ijk} F_{ijk} H_i(w_x) H_j(w_y) H_k(w_z)
   * \f]
   *
   * with scaled random velocity
   *
   * \f[
   * w = \frac{v - u}{\sqrt{2} \sigma},
   * \f]
   *
   * thermal velocity
   *
   * \f[
   * \sigma = \sqrt{P / rho},
   * \f]
   *
   * and scaling coefficient
   *
   * \f[
   * f_0 = \frac{n}{(\sqrt{2 \sqrt{\pi}} \sigma)^3}.
   * \f]
   *
   * The normalized primitive moments are defined
   *
   * \f[
   * \Lambda_{ijk} = \int_{-\infty}^{\infty} w_x^i w_y^j w_z^k f d^3w
   * \f]
   *
   * We convert the hermite coefficients into the normalized primitive moments by replace f in the above expression
   *
   * \f[
   * \Lambda_{ijk}
   * =
   * f_0 \int_{-\infty}^{\infty} w_x^i w_y^j w_z^k \exp(-w^2) \sum_{lmn} F_{lmn} H_l(w_x) H_m(w_y) H_n(w_z) \, d^3w
   * =
   * f_0 \sum_{lmn} M_{il} M_{jm} M_{kn} F_{lmn}
   * \f]
   *
   * where we have defined the "Transform Matrix"
   *
   * \f[
   * M_{ij} = \int_{-\infty}^{\infty} x^i H_j(x) \exp(-x^2) \, dx.
   * \f]
   *
   */

  const unsigned int num_moments = moment_indexer.getNumMoments();
  if(num_moments >= 20) throw std::logic_error("convertNormalizedPrimitiveToHermite : Transform matrix is too small.");
  static const Matrix transform_matrix = buildTransformMatrix(20);

  const double sigma = std::sqrt(P / n / mass);
  const double f0 = n / std::pow(std::sqrt(2. * std::sqrt(M_PI)) * sigma, 3);

  for(unsigned int i=0, pidx=0; i<num_moments; ++i){
    for(unsigned int j=0; j<num_moments-i; ++j){
      for(unsigned int k=0; k<num_moments-i-j; ++k, ++pidx){

        // We are summing over a lot of things so we use a larger double
        KahanSum value;

        // The transformation matrix is lower triangular
        for(unsigned int l=0; l<=i; ++l){
          const double mult_l = transform_matrix(i,l);
          for(unsigned int m=0; m<=j; ++m){
            const double mult_lm = mult_l * transform_matrix(j,m);
            for(unsigned int n=0; n<=k; ++n){
              const double mult_lmn = mult_lm * transform_matrix(k,n);
              value += mult_lmn * hermite_coefficients[moment_indexer(l,m,n)];
            }
          }
        }

        normalized_primitive_moments[pidx] = f0 * value.sum();

      }
    }
  }

}

/**
 * \brief Convert from the Hermite coefficients to the primitive coefficients.
 *
 * \note Primitive coefficients are different from normalized primitive moments!!!
 *
 * \param[in] mass Particle mass
 * \param[in] n Number density
 * \param[in] P Pressure
 * \param[in] moment_indexer Moment indexer for the given moment set
 * \param[in] hermite_coefficients Input hermite coefficients indexed using moment_indexer
 * \param[out] primitive_coefficients Output primitive moments indexed using moment_indexer
 */
inline
void
convertHermiteCoefficientsToPrimitiveCoefficients(const double mass,
                                                  const double n,
                                                  const double P,
                                                  const MomentIndexer & moment_indexer,
                                                  const double * const __restrict__ hermite_coefficients,
                                                  double * const __restrict__ primitive_coefficients)
{

  /**
   *
   * Here we want to convert the Hermite expansion coefficients
   *
   * \f[
   * f = f_0 \exp(-w^2) \sum_{ijk} F_{ijk} H_i(w_x) H_j(w_y) H_k(w_x)
   * \f]
   *
   * to a set of primitive coefficients
   *
   * \f[
   * g = g_0 \exp(-w^2) \sum_{ijk} G_{ijk} w_x^i w_y^i w_x^i
   * \f]
   *
   * Note that we have
   *
   * \f[
   * f_0 = \frac{n}{(\sqrt{2 \sqrt{\pi}} \sigma)^3)
   * \f]
   *
   * and
   *
   * \f[
   * g_0 = \frac{n}{(\sqrt{2 \pi} \sigma)^3).
   * \f]
   *
   * We use a projection to convert from one space to the other
   *
   * \f[
   * \int_{-\infty}^{\infty} (f - g) H_i(w_x) H_j(w_y) H_k(w_z) d^3w = 0
   * \f]
   *
   * which simplifies down to
   *
   * \f[
   * g_0 \sum_{lmn} G_{lmn} M_{li} M_{mj} M_{nk} = f_0 F_{ijk}
   * \f]
   *
   * where we have defined the "Transform Matrix" (yes it shows up everywhere)
   *
   * \f[
   * M_{ij} = \int_{-\infty}^{\infty} x^i H_j(x) \exp(-x^2) dx.
   * \f]
   *
   * We define the inverse transformation matrix \f$ M^{-1}_{ij} \f$ such that
   *
   * \f[
   * \sum_j M^{-1}_{ij} M_{jk} = \sum_j M_{ij} M^{-1}_{jk} = \delta_{ik}
   * \f]
   *
   * which is used to get the Hermite coefficients from the normalized primitive moments
   *
   * \f[
   * G_{ijk} = \frac{f_0}{g_0} \sum_{abc} M^{-1}_{ai} M^{-1}_{bj} M^{-1}_{ck} F_{abc}
   * \f]
   *
   */

  const unsigned int num_moments = moment_indexer.getNumMoments();
  if(num_moments >= 20) throw std::logic_error("convertHermiteCoefficientsToPrimitiveCoefficients : Transform matrix is too small.");
  static const Matrix inverse_transform_matrix = buildInverseTransformMatrix(20);

  const double sigma = std::sqrt(P / n / mass);
  const double hermite_f0 = n / std::pow(std::sqrt(2. * std::sqrt(M_PI)) * sigma, 3);
  const double primitive_f0 = n / std::pow(std::sqrt(2. * M_PI) * sigma, 3);

  const double multiplier = hermite_f0 / primitive_f0;

  for(unsigned int i=0, pidx=0; i<num_moments; ++i){
    for(unsigned int j=0; j<num_moments-i; ++j){
      for(unsigned int k=0; k<num_moments-i-j; ++k,++pidx){

        // We are summing over a lot of things so we use a larger double
        KahanSum value;

        // The transformation matrix is lower triangular
        for(unsigned int a=0, hidx=0; a<num_moments; ++a){
          const double mult_a = inverse_transform_matrix(a,i);
          for(unsigned int b=0; b<num_moments-a; ++b){
            const double mult_ab = mult_a * inverse_transform_matrix(b,j);
            for(unsigned int c=0; c<num_moments-a-b; ++c,++hidx){
              const double mult_abc = mult_ab * inverse_transform_matrix(c,k);
              value += mult_abc * hermite_coefficients[hidx];
            }
          }
        }

        primitive_coefficients[pidx] = multiplier * value.sum();

      }
    }
  }

}

}

#endif
