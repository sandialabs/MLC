
#include "UnitTestHarness.hpp"

#include "MLC_MomentIndexer.hpp"
#include "MLC_MomentConversion.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{

int
factorial(const unsigned int i)
{
  if(i == 0)
    return 1;
  return i * factorial(i-1);
}

double
bc(const unsigned int n,
   const unsigned int k)
{
  if(k > n) return 0.;
  if(k == n) return 1.;
  return static_cast<double>(factorial(n)) / static_cast<double>(factorial(n-k) * factorial(k));
}
}


ADD_TEST(MomentConversion, BinomialCoefficients){

  using namespace mlc;

  auto coeff = buildBinomialCoefficientMatrix(4);

  ASSERT_EQ(coeff.width(), 4);
  ASSERT_EQ(coeff.height(), 4);

  for(int i=0; i<coeff.height(); ++i){
    for(int j=0; j<coeff.width(); ++j){
      const double actual = coeff(i,j);
      const double expected = bc(i,j);
      ASSERT_NEARLY_EQ(actual, expected, 1.e-14);
    }
  }

}


ADD_TEST(MomentConversion, TransformMatrices){

  using namespace mlc;

  const unsigned int size = 12;

  const Matrix A = buildTransformMatrix(size);
  ASSERT_EQ(A.height(), size);
  ASSERT_EQ(A.width(), size);

  const Matrix invA = buildInverseTransformMatrix(size);
  ASSERT_EQ(invA.height(), size);
  ASSERT_EQ(invA.width(), size);

  Matrix result(size,size);
  dot(A,invA,result);

  for(unsigned int i=0; i<size; ++i){
    for(unsigned int j=0; j<size; ++j){
      ASSERT_NEARLY_EQ(result(i,j), double(i==j), 1.e-12);
    }
  }

}

ADD_TEST(MomentConversion, convertConservedMomentsToNormalizedPrimitiveMoments)
{
  using namespace mlc;

  MomentIndexer mi;
  mi.setupForMoments(3);

  std::vector<double> conserved(mi.getNumCoefficients());
  std::vector<double> primitive(mi.getNumCoefficients());

  const double mass = 1.5;
  const double n = 2.1;
  const double u[3] = {-1.1,0.4,0.8};
  const double P0 = 1.5;
  const double P[3][3] = {{P0+0.1,-0.3,0.6},{-0.3,P0+0.4,-0.2},{0.6,-0.2,P0-0.5}};

  // Symmetry and trace test
  ASSERT_NEARLY_EQ(P[0][1], P[1][0], 1.e-14);
  ASSERT_NEARLY_EQ(P[0][2], P[2][0], 1.e-14);
  ASSERT_NEARLY_EQ(P[1][2], P[2][1], 1.e-14);
  ASSERT_NEARLY_EQ(P[0][0]+P[1][1]+P[2][2], 3*P0, 1.e-14);

  // Initialize density
  conserved[mi(0,0,0)] = mass * n;

  // Initialize momentum
  conserved[mi(1,0,0)] = mass * n * u[0];
  conserved[mi(0,1,0)] = mass * n * u[1];
  conserved[mi(0,0,1)] = mass * n * u[2];

  // Initialize energy tensor
  conserved[mi(2,0,0)] = P[0][0] + mass * n * u[0] * u[0];
  conserved[mi(1,1,0)] = P[0][1] + mass * n * u[0] * u[1];
  conserved[mi(1,0,1)] = P[0][2] + mass * n * u[0] * u[2];
  conserved[mi(0,2,0)] = P[1][1] + mass * n * u[1] * u[1];
  conserved[mi(0,1,1)] = P[1][2] + mass * n * u[1] * u[2];
  conserved[mi(0,0,2)] = P[2][2] + mass * n * u[2] * u[2];

  // Convert conserved to the normalize primitive variables
  convertConservedMomentsToNormalizedPrimitiveMoments(mass,n,u[0],u[1],u[2],P0,mi,conserved.data(),primitive.data());

  // Now test the primitive variables
  // Note that the "normalize" in normalized primitive variables refers to the thermal scaling

  // Thermal scaling coefficient
  const double scaled_sigma = std::sqrt(2. * P0 / n / mass);

  // Initialize with density scaling
  double coeff = 1. / std::pow(scaled_sigma,3) / mass;

  // Test density
  ASSERT_NEARLY_EQ(primitive[mi(0,0,0)], coeff * mass * n, 1.e-14);

  // Reset scaling for velocity test
  //coeff = 1. / std::pow(scaled_sigma,4) / mass;

  // Test velocity (note that since the distribution gets shifted to the primitive frame the normalized primitive velocity is zero)
  ASSERT_NEARLY_EQ(primitive[mi(1,0,0)], 0., 1.e-14);
  ASSERT_NEARLY_EQ(primitive[mi(0,1,0)], 0., 1.e-14);
  ASSERT_NEARLY_EQ(primitive[mi(0,0,1)], 0., 1.e-14);

  // Reset scaling for energy test
  coeff = 1. / std::pow(scaled_sigma,5) / mass;

  // Test pressure
  ASSERT_NEARLY_EQ(primitive[mi(2,0,0)], coeff * P[0][0], 1.e-14);
  ASSERT_NEARLY_EQ(primitive[mi(1,1,0)], coeff * P[0][1], 1.e-14);
  ASSERT_NEARLY_EQ(primitive[mi(1,0,1)], coeff * P[0][2], 1.e-14);
  ASSERT_NEARLY_EQ(primitive[mi(0,2,0)], coeff * P[1][1], 1.e-14);
  ASSERT_NEARLY_EQ(primitive[mi(0,1,1)], coeff * P[1][2], 1.e-14);
  ASSERT_NEARLY_EQ(primitive[mi(0,0,2)], coeff * P[2][2], 1.e-14);

}

ADD_TEST(MomentConversion, convertNormalizedPrimitiveMomentsToHermiteCoefficients)
{

  using namespace mlc;

  // This test is generated in mathematica and pulled into C++ using some truly terrible scripting...

  // From mathematica
  const double mass = 1.4;
  const double rho = 2.94;
  const double u[3] = {-1.2, 5.2, 2.6};
  const double sigma = 1.3;

  const double n = rho / mass;
  const double P = sigma * sigma * rho;

  // While this is required for the math, it can be set to anything (cancels with itself)
  const double kB = 2.2;

  const unsigned int num_moments = 3;
  MomentIndexer mi;
  mi.setupForMoments(num_moments);

  // Hermite coefficients (expected)
  std::vector<double> expected_hermite_coefficients(mi.getNumCoefficients());
  {

    // Mathematica output
    double F[3][3][3];
    F[0][0][0] = 1.0000000000000000000000000000000;
    F[0][0][1] = 0;
    F[0][0][2] = -0.82945172399142356936955247812802;
    F[0][1][0] = 0;
    F[0][1][1] = 0.034476854128837919640956676298680;
    F[0][1][2] = 0.68441973070019236734978781876072;
    F[0][2][0] = 0.75201742518893599125913227698082;
    F[0][2][1] = 0.50598388004697276094684736342280;
    F[0][2][2] = 0.17417048699738669543311396859842;
    F[1][0][0] = 0;
    F[1][0][1] = -0.27580876158455811420886559574728;
    F[1][0][2] = -0.95039846190677123310771738723154;
    F[1][1][0] = -0.31340953513300921468522006383837;
    F[1][1][1] = 0.90939453214999733027438870618835;
    F[1][1][2] = 0.46729776571178718467730226145521;
    F[1][2][0] = -0.55932894632219056404354760400986;
    F[1][2][1] = -0.59536130211861653649456998917864;
    F[1][2][2] = 0.40516597653185198667601546182505;
    F[2][0][0] = 0.077434298802487578110420201147197;
    F[2][0][1] = -0.69279435322732823059378162708600;
    F[2][0][2] = -0.50000644726086458455693446168714;
    F[2][1][0] = -0.034773148563910002578294700687958;
    F[2][1][1] = -0.84718356048294744494557710858918;
    F[2][1][2] = 0.43552418655695721876304092720444;
    F[2][2][0] = -0.078071807327745872620735644685241;
    F[2][2][1] = 0.87810641223003413381251856074720;
    F[2][2][2] = 0.39436093792690638994895897132553;

    for(unsigned int i=0, idx=0; i<num_moments; ++i)
      for(unsigned int j=0; j<num_moments-i; ++j)
        for(unsigned int k=0; k<num_moments-i-j; ++k,++idx)
          expected_hermite_coefficients[idx] = F[i][j][k];


  }

  // Conserved moments associated with above polynomials - not sure why the values are truncated...
  std::vector<double> conserved_moments(mi.getNumCoefficients());
  {

    // Mathematica output - not sure why the values are truncated... means we will have to adjust our relative error
    double F[3][3][3];
    F[0][0][0] = 2.94;
    F[0][0][1] = 7.644;
    F[0][0][2] = 19.0147;
    F[0][1][0] = 15.288;
    F[0][1][1] = 39.9201;
    F[0][1][2] = 106.019;
    F[0][2][0] = 89.7504;
    F[0][2][1] = 239.754;
    F[0][2][2] = 692.188;
    F[1][0][0] = -3.528;
    F[1][0][1] = -10.5432;
    F[1][0][2] = -38.6252;
    F[1][1][0] = -19.9028;
    F[1][1][1] = -53.2049;
    F[1][1][2] = -186.487;
    F[1][2][0] = -129.005;
    F[1][2][1] = -328.448;
    F[1][2][2] = -1117.34;
    F[2][0][0] = 9.7463;
    F[2][0][1] = 22.3009;
    F[2][0][2] = 60.747;
    F[2][1][0] = 54.1004;
    F[2][1][1] = 101.234;
    F[2][1][2] = 237.707;
    F[2][2][0] = 343.066;
    F[2][2][1] = 609.588;
    F[2][2][2] = 1431.41;

    for(unsigned int i=0, idx=0; i<num_moments; ++i)
      for(unsigned int j=0; j<num_moments-i; ++j)
        for(unsigned int k=0; k<num_moments-i-j; ++k,++idx)
          conserved_moments[idx] = F[i][j][k];
  }

  // Generate the normalized primitive coefficients from the conserved variables
  std::vector<double> normalized_primitive_moments(mi.getNumCoefficients());
  convertConservedMomentsToNormalizedPrimitiveMoments(mass, n, u[0], u[1], u[2], P, mi, conserved_moments.data(), normalized_primitive_moments.data());

  // Generate the hermite polynomials from the conserved moments
  std::vector<double> actual_hermite_coefficients(mi.getNumCoefficients());
  convertNormalizedPrimitiveMomentsToHermiteCoefficients(mass, n, P, mi, normalized_primitive_moments.data(), actual_hermite_coefficients.data());

  // Compare what we got for the hermite coefficients against the Mathematica output
  for(unsigned int i=0,idx=0; i<3u; ++i)
    for(unsigned int j=0; j<num_moments-i; ++j)
      for(unsigned int k=0; k<num_moments-i-j; ++k,++idx){
        const double actual_Fijk = actual_hermite_coefficients[idx];
        const double expected_Fijk = expected_hermite_coefficients[idx];
        ASSERT_NEARLY_EQ(actual_Fijk, expected_Fijk, 4.e-4);
      }

}


ADD_TEST(MomentConversion, convertHermiteCoefficientsToPrimitiveCoefficients)
{
  using namespace mlc;

  const unsigned int num_moments = 3;
  MomentIndexer mi;
  mi.setupForMoments(num_moments);

  std::vector<double> conserved(mi.getNumCoefficients());
  std::vector<double> normalized_primitive(mi.getNumCoefficients());
  std::vector<double> hermite(mi.getNumCoefficients());
  std::vector<double> primitive(mi.getNumCoefficients());

  const double mass = 1.5;
  const double n = 2.1;
  const double u[3] = {-1.1,0.4,0.8};
  const double P0 = 1.5;
  const double P[3][3] = {{P0+0.1,-0.3,0.6},{-0.3,P0+0.4,-0.2},{0.6,-0.2,P0-0.5}};

  const double sigma = std::sqrt(P0 / n / mass);
  const double primitive_f0 = n / std::pow(2. * M_PI * sigma * sigma, 1.5);
  const double hermite_f0 = n / std::pow(2. * std::sqrt(M_PI) * sigma * sigma, 1.5);

  // Symmetry and trace test
  ASSERT_NEARLY_EQ(P[0][1], P[1][0], 1.e-14);
  ASSERT_NEARLY_EQ(P[0][2], P[2][0], 1.e-14);
  ASSERT_NEARLY_EQ(P[1][2], P[2][1], 1.e-14);
  ASSERT_NEARLY_EQ(P[0][0]+P[1][1]+P[2][2], 3*P0, 1.e-14);

  // Initialize density
  conserved[mi(0,0,0)] = mass * n;

  // Initialize momentum
  conserved[mi(1,0,0)] = mass * n * u[0];
  conserved[mi(0,1,0)] = mass * n * u[1];
  conserved[mi(0,0,1)] = mass * n * u[2];

  // Initialize energy tensor
  conserved[mi(2,0,0)] = P[0][0] + mass * n * u[0] * u[0];
  conserved[mi(1,1,0)] = P[0][1] + mass * n * u[0] * u[1];
  conserved[mi(1,0,1)] = P[0][2] + mass * n * u[0] * u[2];
  conserved[mi(0,2,0)] = P[1][1] + mass * n * u[1] * u[1];
  conserved[mi(0,1,1)] = P[1][2] + mass * n * u[1] * u[2];
  conserved[mi(0,0,2)] = P[2][2] + mass * n * u[2] * u[2];

  // Convert conserved to the normalize primitive variables
  convertConservedMomentsToNormalizedPrimitiveMoments(mass,n,u[0],u[1],u[2],P0,mi,conserved.data(),normalized_primitive.data());

  // Test the primitive moments
  {

    // Initialize with density scaling
    double coeff = 1. / std::pow(M_SQRT2 * sigma,3) / mass;

    // Test density
    ASSERT_NEARLY_EQ(normalized_primitive[mi(0,0,0)], coeff * mass * n, 1.e-14);

    // Reset scaling for velocity test
    //coeff = 1. / std::pow(scaled_sigma,4) / mass;

    // Test velocity (note that since the distribution gets shifted to the primitive frame the normalized primitive velocity is zero)
    ASSERT_NEARLY_EQ(normalized_primitive[mi(1,0,0)], 0., 1.e-14);
    ASSERT_NEARLY_EQ(normalized_primitive[mi(0,1,0)], 0., 1.e-14);
    ASSERT_NEARLY_EQ(normalized_primitive[mi(0,0,1)], 0., 1.e-14);

    // Reset scaling for energy test
    coeff = 1. / std::pow(M_SQRT2 * sigma,5) / mass;

    // Test pressure
    ASSERT_NEARLY_EQ(normalized_primitive[mi(2,0,0)], coeff * P[0][0], 1.e-14);
    ASSERT_NEARLY_EQ(normalized_primitive[mi(1,1,0)], coeff * P[0][1], 1.e-14);
    ASSERT_NEARLY_EQ(normalized_primitive[mi(1,0,1)], coeff * P[0][2], 1.e-14);
    ASSERT_NEARLY_EQ(normalized_primitive[mi(0,2,0)], coeff * P[1][1], 1.e-14);
    ASSERT_NEARLY_EQ(normalized_primitive[mi(0,1,1)], coeff * P[1][2], 1.e-14);
    ASSERT_NEARLY_EQ(normalized_primitive[mi(0,0,2)], coeff * P[2][2], 1.e-14);
  }

  // Convert normalized primitive to hermite
  convertNormalizedPrimitiveMomentsToHermiteCoefficients(mass, n, P0, mi, normalized_primitive.data(), hermite.data());

  {

    /**
     * The hermite coefficients for this system can be defined using the transform matrix from the primitive moments.
     *
     * We know the definition of the pressure
     *
     * \f[
     * P_{ij} = m (\sqrt{2} \sigma)^2 \int_{-\infty}^{\infty} w_i w_j f \, d^3v.
     * \f]
     *
     * We can apply our primitive expansion
     *
     * \f[
     * P_{ij} = m f_0 (\sqrt{2} \sigma)^5 \int_{-\infty}^{\infty} w_i w_j \exp(-w^2) \sum_{lmn} F_{lmn} H_l(w_x) H_m(w_y) H_n(w_z) \, d^3w
     * \f]
     *
     * If we define the integral
     *
     * \f[
     * M_{ij} = \int_{-\infty}^{\infty} x^i H_j(x) \exp(-x^2) \, dx
     * \f]
     *
     * we can simplify our pressure relation to
     *
     * \f[
     * P_{xx} &= m f_0 (\sqrt{2} \sigma)^5 \sum_{ijk} F_{ijk} M_{2i} M_{0j} M_{0k} \\
     * P_{xy} &= m f_0 (\sqrt{2} \sigma)^5 \sum_{ijk} F_{ijk} M_{1i} M_{1j} M_{0k} \\
     * P_{xz} &= m f_0 (\sqrt{2} \sigma)^5 \sum_{ijk} F_{ijk} M_{1i} M_{0j} M_{1k} \\
     * P_{yy} &= m f_0 (\sqrt{2} \sigma)^5 \sum_{ijk} F_{ijk} M_{0i} M_{2j} M_{0k} \\
     * P_{yz} &= m f_0 (\sqrt{2} \sigma)^5 \sum_{ijk} F_{ijk} M_{0i} M_{1j} M_{1k} \\
     * P_{zz} &= m f_0 (\sqrt{2} \sigma)^5 \sum_{ijk} F_{ijk} M_{0i} M_{0j} M_{2k}.
     * \f]
     *
     * We use this to define our test
     *
     * \f[
     * \sum_{ijk} F_{ijk} M_{2i} M_{0j} M_{0k} = \frac{P_{xx}}{m f_0 (\sqrt{2} \sigma)^5} \\
     * \sum_{ijk} F_{ijk} M_{1i} M_{1j} M_{0k} = \frac{P_{xy}}{m f_0 (\sqrt{2} \sigma)^5} \\
     * \sum_{ijk} F_{ijk} M_{1i} M_{0j} M_{1k} = \frac{P_{xz}}{m f_0 (\sqrt{2} \sigma)^5} \\
     * \sum_{ijk} F_{ijk} M_{0i} M_{2j} M_{0k} = \frac{P_{yy}}{m f_0 (\sqrt{2} \sigma)^5} \\
     * \sum_{ijk} F_{ijk} M_{0i} M_{1j} M_{1k} = \frac{P_{yz}}{m f_0 (\sqrt{2} \sigma)^5} \\
     * \sum_{ijk} F_{ijk} M_{0i} M_{0j} M_{2k} = \frac{P_{zz}}{m f_0 (\sqrt{2} \sigma)^5}
     * \f]
     *
     */

    const Matrix M = buildTransformMatrix(num_moments);

    // Test density - If n,u,P are used to define the conserved variables, then the 000 coefficient must be 1
    ASSERT_NEARLY_EQ(hermite[mi(0,0,0)], 1.0, 1.e-14);

    // Test velocity - these should be zero since the system is now centered at the average velocity
    ASSERT_NEARLY_EQ(hermite[mi(1,0,0)], 0.0, 1.e-14);
    ASSERT_NEARLY_EQ(hermite[mi(0,1,0)], 0.0, 1.e-14);
    ASSERT_NEARLY_EQ(hermite[mi(0,0,1)], 0.0, 1.e-14);

    // Test pressure

    // This is the LHS of the test
    double scaled_pressure[6] = {0};
    for(unsigned int i=0, idx=0; i<num_moments; ++i){
      for(unsigned int j=0; j<num_moments-i; ++j){
        for(unsigned int k=0; k<num_moments-i-j; ++k,++idx){
          out << "Hermite Coefficient ("<<i<<", "<<j<<", "<<k<<"): " << hermite[idx] << "\n";
          scaled_pressure[0] += hermite[idx] * M(2,i) * M(0,j) * M(0,k);
          scaled_pressure[1] += hermite[idx] * M(1,i) * M(1,j) * M(0,k);
          scaled_pressure[2] += hermite[idx] * M(1,i) * M(0,j) * M(1,k);
          scaled_pressure[3] += hermite[idx] * M(0,i) * M(2,j) * M(0,k);
          scaled_pressure[4] += hermite[idx] * M(0,i) * M(1,j) * M(1,k);
          scaled_pressure[5] += hermite[idx] * M(0,i) * M(0,j) * M(2,k);
        }
      }
    }

    const double mult = 1. / (mass * hermite_f0 * std::pow(M_SQRT2 * sigma,5));
    ASSERT_NEARLY_EQ(scaled_pressure[0], mult * P[0][0], 1.e-14);
    ASSERT_NEARLY_EQ(scaled_pressure[1], mult * P[0][1], 1.e-14);
    ASSERT_NEARLY_EQ(scaled_pressure[2], mult * P[0][2], 1.e-14);
    ASSERT_NEARLY_EQ(scaled_pressure[3], mult * P[1][1], 1.e-14);
    ASSERT_NEARLY_EQ(scaled_pressure[4], mult * P[1][2], 1.e-14);
    ASSERT_NEARLY_EQ(scaled_pressure[5], mult * P[2][2], 1.e-14);
  }

  // Convert hermite coefficients to primitive coefficients
  convertHermiteCoefficientsToPrimitiveCoefficients(mass, n, P0, mi, hermite.data(), primitive.data());

  // Test the primitive coefficients
  {

    /**
     * The normalized form for the density and velocity are trivial (1,0,0,0), however the pressure tensor is more challenging
     *
     * We know the definition of the pressure
     *
     * \f[
     * P_{ij} = m (\sqrt{2} \sigma)^2 \int_{-\infty}^{\infty} w_i w_j f \, d^3v.
     * \f]
     *
     * We can apply our primitive expansion
     *
     * \f[
     * P_{ij} = m f_0 (\sqrt{2} \sigma)^5 \int_{-\infty}^{\infty} w_i w_j \exp(-w^2) \sum_{lmn} F_{lmn} w_x^l w_y^m w_z^n \, d^3w
     * \f]
     *
     * If we define the integral
     *
     * \f[
     * I_i = (1/\sqrt{\pi}) \int_{-\infty}^{\infty} x^i \exp(-x^2) \, dx
     * \f]
     *
     * we can simplify our pressure relation to
     *
     * \f[
     * P_{xx} &= m f_0 \pi^{3/2} (\sqrt{2} \sigma)^5 \sum_{ijk} F_{ijk} I_{i+2} I_{j  } I_{k  } \\
     * P_{xy} &= m f_0 \pi^{3/2} (\sqrt{2} \sigma)^5 \sum_{ijk} F_{ijk} I_{i+1} I_{j+1} I_{k  } \\
     * P_{xz} &= m f_0 \pi^{3/2} (\sqrt{2} \sigma)^5 \sum_{ijk} F_{ijk} I_{i+1} I_{j  } I_{k+1} \\
     * P_{yy} &= m f_0 \pi^{3/2} (\sqrt{2} \sigma)^5 \sum_{ijk} F_{ijk} I_{i  } I_{j+2} I_{k  } \\
     * P_{yz} &= m f_0 \pi^{3/2} (\sqrt{2} \sigma)^5 \sum_{ijk} F_{ijk} I_{i  } I_{j+1} I_{k+1} \\
     * P_{zz} &= m f_0 \pi^{3/2} (\sqrt{2} \sigma)^5 \sum_{ijk} F_{ijk} I_{i  } I_{j  } I_{k+2}.
     * \f]
     *
     * We use this to define our test
     *
     * \f[
     * \sum_{ijk} F_{ijk} I_{i+2} I_{j  } I_{k  } = \frac{P_{xx}}{m f_0 \pi^{3/2} (\sqrt{2} \sigma)^5} \\
     * \sum_{ijk} F_{ijk} I_{i+1} I_{j+1} I_{k  } = \frac{P_{xy}}{m f_0 \pi^{3/2} (\sqrt{2} \sigma)^5} \\
     * \sum_{ijk} F_{ijk} I_{i+1} I_{j  } I_{k+1} = \frac{P_{xz}}{m f_0 \pi^{3/2} (\sqrt{2} \sigma)^5} \\
     * \sum_{ijk} F_{ijk} I_{i  } I_{j+2} I_{k  } = \frac{P_{yy}}{m f_0 \pi^{3/2} (\sqrt{2} \sigma)^5} \\
     * \sum_{ijk} F_{ijk} I_{i  } I_{j+1} I_{k+1} = \frac{P_{yz}}{m f_0 \pi^{3/2} (\sqrt{2} \sigma)^5} \\
     * \sum_{ijk} F_{ijk} I_{i  } I_{j  } I_{k+2} = \frac{P_{zz}}{m f_0 \pi^{3/2} (\sqrt{2} \sigma)^5}
     * \f]
     *
     */

    // Test density - If n,u,P are used to define the conserved variables, then the 000 coefficient must be 1
    ASSERT_NEARLY_EQ(primitive[mi(0,0,0)], 1.0, 1.e-14);

    // Test velocity - these should be zero since the system is now centered at the average velocity
    ASSERT_NEARLY_EQ(primitive[mi(1,0,0)], 0.0, 1.e-14);
    ASSERT_NEARLY_EQ(primitive[mi(0,1,0)], 0.0, 1.e-14);
    ASSERT_NEARLY_EQ(primitive[mi(0,0,1)], 0.0, 1.e-14);

    // Test pressure
    // I_i = (1/\sqrt{\pi}) \int_{-\infty}^{\infty} x^i \exp(-x^2) dx
    const double I[8] = {1., 0., 0.5, 0., 0.75, 0., 15./8., 0.};

    // This is the LHS of the test
    double scaled_pressure[6] = {0};
    for(unsigned int i=0, idx=0; i<num_moments; ++i){
      for(unsigned int j=0; j<num_moments-i; ++j){
        for(unsigned int k=0; k<num_moments-i-j; ++k,++idx){
          out << "Primitive Coefficient ("<<i<<", "<<j<<", "<<k<<"): " << primitive[idx] << "\n";
          scaled_pressure[0] += primitive[idx] * I[i+2] * I[j  ] * I[k  ];
          scaled_pressure[1] += primitive[idx] * I[i+1] * I[j+1] * I[k  ];
          scaled_pressure[2] += primitive[idx] * I[i+1] * I[j  ] * I[k+1];
          scaled_pressure[3] += primitive[idx] * I[i  ] * I[j+2] * I[k  ];
          scaled_pressure[4] += primitive[idx] * I[i  ] * I[j+1] * I[k+1];
          scaled_pressure[5] += primitive[idx] * I[i  ] * I[j  ] * I[k+2];
        }
      }
    }

    const double mult = 1. / (mass * std::pow(M_PI,1.5) * primitive_f0 * std::pow(M_SQRT2 * sigma,5));
    ASSERT_NEARLY_EQ(scaled_pressure[0], mult * P[0][0], 1.e-14);
    ASSERT_NEARLY_EQ(scaled_pressure[1], mult * P[0][1], 1.e-14);
    ASSERT_NEARLY_EQ(scaled_pressure[2], mult * P[0][2], 1.e-14);
    ASSERT_NEARLY_EQ(scaled_pressure[3], mult * P[1][1], 1.e-14);
    ASSERT_NEARLY_EQ(scaled_pressure[4], mult * P[1][2], 1.e-14);
    ASSERT_NEARLY_EQ(scaled_pressure[5], mult * P[2][2], 1.e-14);
  }

}
