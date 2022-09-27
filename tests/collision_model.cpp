
#include "UnitTestHarness.hpp"

#include "MLC_CollisionModel.hpp"
#include "MLC_MomentConversion.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>

double drand(double min, double max)
{
    double f = (double) rand() / RAND_MAX;
    return min + f * (max - min);
}

/**

To recreate the model used for this call, there are three calls:

1) Generate data

./example_model_data_generator.py

This creates the files example_model_training.h5 and example_model_testing.h5, which store training and testing data.
You'll need to copy these data files into a directory with the example_model_input.yaml file.
In this directory, you will then call the single species build_model.py script:

.../fluid-kinetic/codes/model_builder/single_species/build_model.py example_model_input.yaml

This will load the data, generate the model, then train the model and output the diffs associated with the testing data.
The resulting model directory "example_model" is what gets placed in this test directory.

 */

ADD_TEST(CollisionModel, ModelEvaluation)
{
  using namespace mlc;

  const unsigned int num_moments = 4;

  const std::string model_name = "example_model";

  const unsigned int num_points = 10;

  // Note that there are 20 total moments, but the ML model only reads in 15
  const unsigned int num_coefficients = 15;

  CollisionModel model(num_points, model_name);

  MLArray n({num_points});
  MLArray T({num_points});
  MLArray F({num_points, num_coefficients});
  MLArray expected_tau({num_points});
  MLArray expected_G({num_points, num_coefficients});

  const double n_span[2] = {2.0,4.0};
  const double T_span[2] = {8.0,12.0};
  const double F_span[2] = {-2.0,2.0};

  for(unsigned int point = 0; point < num_points; ++point){
    n[point] = drand(n_span[0], n_span[1]);
    T[point] = drand(T_span[0], T_span[1]);

    expected_tau[point] = T[point] / n[point];

    unsigned int idx=0;
    for(unsigned int i=0; i<num_moments; ++i){
      for(unsigned int j=0; j<num_moments-i; ++j){
        for(unsigned int k=0; k<num_moments-i-j; ++k){
          if((i+j+k<2) or ((j+k)==0 and i==2))
            continue;

          const double f = drand(F_span[0],F_span[1]);
          F[point*num_coefficients + idx] = f;
          expected_G[point*num_coefficients + idx] = i * n[point] + j * T[point] + k * f;

          ++idx;
        }
      }
    }
  }

  MLArray actual_tau({num_points});
  MLArray actual_G({num_points, num_coefficients});

  model.evaluateModel(n,T,F,actual_tau,actual_G);

  const double tolerance = 0.03;

  for(unsigned int point=0; point<num_points; ++point){

    out << "Point "<<point<<":\n";
    out << "\tExpected tau: " << expected_tau[point] << "\n";
    out << "\tActual tau: " << actual_tau[point] << "\n";

    const double rel_tau = std::fabs((expected_tau[point] - actual_tau[point]) / actual_tau[point]);

    out << "\tRelative tau error: " << rel_tau << "\n";
    ASSERT_LT(rel_tau, tolerance);

    out << "\tExpected G: ";
    for(unsigned int i=0; i<num_coefficients; ++i)
      out << std::setw(14) << expected_G[point*num_coefficients+i] << " ";
    out << "\n";
    out << "\tActual G:   ";
    for(unsigned int i=0; i<num_coefficients; ++i)
      out << std::setw(14) << actual_G[point*num_coefficients + i] << " ";
    out << "\n";

    double error = 0, abs_sum = 0;
    for(unsigned int i=0; i<num_coefficients; ++i){
      error += std::fabs(expected_G[point*num_coefficients + i] - actual_G[point*num_coefficients + i]);
      abs_sum += std::fabs(expected_G[point*num_coefficients + i]);
    }
    out << "\tG Relative Error: " << error / abs_sum << "\n";
    ASSERT_LT(error / abs_sum, tolerance);

  }

}

ADD_TEST(CollisionModel, CollisionEvaluation)
{
  using namespace mlc;

  /*

  So to avoid doing actual physics, we are going to instead solve this problem from the middle.
  Basically we have two parts to evaluating a collision operator:

  conserved moments -> input hermite coefficients               (pre-processing in CollisionModel)
  input hermite coefficients -> output hermite coefficients     (neural network evaluation)

  Now in theory, you could figure out the mapping from conserved moments to output hermite coefficients, but that would be a PITA.
  Instead, we are going to start by building the input hermite coefficients and use them to generate both the output
  hermite coefficients (known by the function used to generate the training data - see the previous test), and the conserved moments
  which is an ugly transformation, but a heck of a lot less work than the alternative.

   */

  const double tolerance = 0.03;

  const unsigned int num_moments = 4;

  const std::string model_name = "example_model";

  const unsigned int num_points = 3;

  // We only test the hermite polynomial output
  const bool evaluate_primitive = false;

  MomentIndexer moment_indexer;
  moment_indexer.setupForMoments(num_moments);

  // Note that there are 20 total moments
  const unsigned int num_total_moments = moment_indexer.getNumCoefficients();

  // The ML model has a different moment indexer
  MomentIndexer ml_moment_indexer;
  ml_moment_indexer.setupForCoefficients(num_total_moments-5);

  // There should be 15 ML coefficients
  const unsigned int num_ml_coefficients = ml_moment_indexer.getNumCoefficients();

  // Used to convert between ML coefficients and spectral coefficients
  std::vector<unsigned int> ml_to_moment_index(num_ml_coefficients);
  for(unsigned int i=0; i<num_moments; ++i)
    for(unsigned int j=0; j<num_moments-i; ++j)
      for(unsigned int k=0; k<num_moments-i-j; ++k){
        int idx = ml_moment_indexer(i,j,k);
        if(idx >= 0)
          ml_to_moment_index[idx] = moment_indexer(i,j,k);
      }

  CollisionModel model(num_points, model_name);

  // The model defines these coefficients (important since the model internal converts things to number density and temperature)
  const double kB = model.getBoltzmannConstant();
  const double mass = model.getMass();

  Array conserved_moments({num_points,num_total_moments});
  Array expected_tau({num_points});
  Array expected_G({num_points,num_total_moments});

  Array expected_g0({num_points});
  Array expected_ux({num_points});
  Array expected_uy({num_points});
  Array expected_uz({num_points});
  Array expected_sigma({num_points});

  // These are the bounds associated with the model
  const double n_span[2] = {2.0,4.0};
  const double T_span[2] = {8.0,12.0};
  const double F_span[2] = {-2.0,2.0};

  // We need to test the evaluate call directly
  MLArray ml_F({num_points,num_ml_coefficients});
  MLArray ml_n({num_points});
  MLArray ml_T({num_points});
  MLArray ml_G({num_points,num_ml_coefficients});

  out << "--------------------------------------------------------------------------\n";
  out << "----------------------------- Initialization -----------------------------\n";
  out << "--------------------------------------------------------------------------\n";

  for(unsigned int point = 0; point < num_points; ++point){

    // Define primitive state of fluid
    const double n = drand(n_span[0], n_span[1]);
    const double T = drand(T_span[0], T_span[1]);
    expected_ux[point] = drand(-0.5, 0.5);
    expected_uy[point] = drand(-0.5, 0.5);
    expected_uz[point] = drand(-0.5, 0.5);

    // Define the pressure
    const double P = n * kB * T;

    expected_sigma[point] = std::sqrt(P / n / mass);
    expected_tau[point] = T / n;

    // Build conserved moments and output hermite coefficients
    {

      // Build F (input hermite coefficients) and expected G (output hermite coefficients)
      Array F({num_total_moments});

      // Fill F
      unsigned int fidx=0;
      for(unsigned int i=0; i<num_moments; ++i){
        for(unsigned int j=0; j<num_moments-i; ++j){
          for(unsigned int k=0; k<num_moments-i-j; ++k){

            const double f = drand(F_span[0],F_span[1]);
            F[fidx++] = f;
          }
        }
      }

      // Something required for the moment conversion process
      F[moment_indexer(0,0,0)] = 1.;
      F[moment_indexer(1,0,0)] = 0;
      F[moment_indexer(0,1,0)] = 0;
      F[moment_indexer(0,0,1)] = 0;
      F[moment_indexer(2,0,0)] = - (F[moment_indexer(0,2,0)] + F[moment_indexer(0,0,2)]);

      // Fill G (this is the formula used in the data generation python script)
      for(unsigned int i=0,idx=0; i<num_moments; ++i)
        for(unsigned int j=0; j<num_moments-i; ++j)
          for(unsigned int k=0; k<num_moments-i-j; ++k,++idx)
            expected_G[point*num_total_moments+idx] = i * n + j * T + k * F[idx];

      // Something applied automatically in the CollisionModel call
      expected_G[point*num_total_moments+moment_indexer(0,0,0)] = 0;
      expected_G[point*num_total_moments+moment_indexer(1,0,0)] = 0;
      expected_G[point*num_total_moments+moment_indexer(0,1,0)] = 0;
      expected_G[point*num_total_moments+moment_indexer(0,0,1)] = 0;
      expected_G[point*num_total_moments+moment_indexer(2,0,0)] = - (expected_G[point*num_total_moments+moment_indexer(0,2,0)] + expected_G[point*num_total_moments+moment_indexer(0,0,2)]);

      // Convert F to conserved_moments using normalized primitive moments
      Array normalized_primitive_moments({num_total_moments});

      // First we convert F to normalized primitive moments
      convertHermiteCoefficientsToNormalizedPrimitiveMoments(mass, n, P, moment_indexer, F.data(), normalized_primitive_moments.data());

      // Second we convert the normalized primitive moments into the conserved moments
      convertNormalizedPrimitiveMomentsToConservedMoments(mass, n, expected_ux[point], expected_uy[point], expected_uz[point], P, moment_indexer, normalized_primitive_moments.data(), conserved_moments.data() + point*num_total_moments);

      out << "Point "<<point<<":\n";
      out << "\tmass: " << mass << "\n";
      out << "\tn: " << n << "\n";
      out << "\tux: " << expected_ux[point] << "\n";
      out << "\tuy: " << expected_uy[point] << "\n";
      out << "\tuz: " << expected_uz[point] << "\n";
      out << "\tP: " << P << "\n";
      out << "\tsigma: " << expected_sigma[point] << "\n";
      out << "\ttau: " << expected_tau[point] << "\n";
      out << "\tConserved Moments: ";
      for(unsigned int i=0; i<num_total_moments; ++i)
        out << conserved_moments[point*num_total_moments + i] << " ";
      out << "\n";
      out << "\tF: ";
      for(unsigned int i=0; i<num_total_moments; ++i)
        out << F[i] << " ";
      out << "\n";
      out << "\tG: ";
      for(unsigned int i=0; i<num_total_moments; ++i)
        out << expected_G[point*num_total_moments + i] << " ";
      out << "\n";

      // Test to make sure we did the conversion correctly (NPM)
      {
        Array actual_npm({num_total_moments});
        convertConservedMomentsToNormalizedPrimitiveMoments(mass, n, expected_ux[point], expected_uy[point], expected_uz[point], P, moment_indexer, conserved_moments.data() + point*num_total_moments, actual_npm.data());

        out << "\tNormalized Primitive Moments: ";
        for(unsigned int i=0; i<num_total_moments; ++i)
          out << std::setw(14) << normalized_primitive_moments[i] << " ";
        out << "\n";

        out << "\tactual_npm:                   ";
        for(unsigned int i=0; i<num_total_moments; ++i)
          out << std::setw(14) << actual_npm[i] << " ";
        out << "\n";

        double error = 0, abs_sum = 0;
        for(unsigned int i=0; i<num_total_moments; ++i){
          error += std::fabs(normalized_primitive_moments[i] - actual_npm[i]);
          abs_sum += std::fabs(normalized_primitive_moments[i]);
        }
        out << "\tNormalized Primitive Moments Relative Error: " << error / abs_sum << "\n";
        ASSERT_LT(error / abs_sum, 1.e-8);
      }

      // Test to make sure we did the conversion correctly (HC)
      {
        Array actual_hc({num_total_moments});
        convertNormalizedPrimitiveMomentsToHermiteCoefficients(mass, n, P, moment_indexer, normalized_primitive_moments.data(), actual_hc.data());

        out << "\tHermite Coefficients: ";
        for(unsigned int i=0; i<num_total_moments; ++i)
          out << std::setw(14) << F[i] << " ";
        out << "\n";

        out << "\tactual_hc:            ";
        for(unsigned int i=0; i<num_total_moments; ++i)
          out << std::setw(14) << actual_hc[i] << " ";
        out << "\n";

        double error = 0, abs_sum = 0;
        for(unsigned int i=0; i<num_total_moments; ++i){
          error += std::fabs(F[i] - actual_hc[i]);
          abs_sum += std::fabs(F[i]);
        }
        out << "\tHermite Coefficients Relative Error: " << error / abs_sum << "\n";
        ASSERT_LT(error / abs_sum, 1.e-8);
      }

      // Fill ML values
      ml_n[point] = n;
      ml_T[point] = T;
      for(int i=0; i<num_ml_coefficients; ++i){
        ml_F[point*num_ml_coefficients+i] = F[ml_to_moment_index[i]];
        ml_G[point*num_ml_coefficients+i] = expected_G[point * num_total_moments + ml_to_moment_index[i]];
      }

      out << "\n====================================\n\n";

    }
  }

  out << "-------------------------------------------------------------------------\n";
  out << "----------------------------- Test ML Model -----------------------------\n";
  out << "-------------------------------------------------------------------------\n";

  // Now make sure that the internal ML model can recreate G
  {

    // Evaluate the model
    MLArray ml_actual_tau({num_points}), ml_actual_G({num_points,num_ml_coefficients});
    model.evaluateModel(ml_n, ml_T, ml_F, ml_actual_tau, ml_actual_G);

    // Check the output
    for(int point=0; point<num_points; ++point){
      out << "Point "<<point<<":\n";
      out << "\tExpected tau: " << expected_tau[point] << "\n";
      out << "\tActual tau:   " << ml_actual_tau[point] << "\n";
      out << "\tExpected G: ";
      for(unsigned int i=0; i<num_ml_coefficients; ++i)
        out << std::setw(14) << ml_G[point*num_ml_coefficients+i] << " ";
      out << "\n";
      out << "\tActual G:   ";
      for(unsigned int i=0; i<num_ml_coefficients; ++i)
        out << std::setw(14) << ml_actual_G[point*num_ml_coefficients + i] << " ";
      out << "\n";

      const double tau_rel_error = std::fabs(expected_tau[point] - ml_actual_tau[point]) / std::fabs(expected_tau[point]);
      out << "\ttau Relative Error: " << tau_rel_error << "\n";
      ASSERT_LT(tau_rel_error, 1.e-2);

      double error = 0, abs_sum = 0;
      for(unsigned int i=0; i<num_ml_coefficients; ++i){
        error += std::fabs(ml_G[point*num_ml_coefficients + i] - ml_actual_G[point*num_ml_coefficients + i]);
        abs_sum += std::fabs(ml_G[point*num_ml_coefficients + i]);
      }
      out << "\tCollision Respeonse Relative Error: " << error / abs_sum << "\n";
      ASSERT_LT(error / abs_sum, 1.e-2);

      out << "\n====================================\n\n";
    }

  }

  out << "--------------------------------------------------------------------------------\n";
  out << "----------------------------- Test Collision Model -----------------------------\n";
  out << "--------------------------------------------------------------------------------\n";

  Array actual_tau({num_points});
  Array actual_G({num_points, num_total_moments});

  Array actual_g0({num_points});
  Array actual_ux({num_points});
  Array actual_uy({num_points});
  Array actual_uz({num_points});
  Array actual_sigma({num_points});

  model.generateCollisionalResponse(conserved_moments,
                                    actual_tau, actual_g0, actual_ux, actual_uy, actual_uz, actual_sigma, actual_G,
                                    evaluate_primitive);

#define CHECK_POINT_VALUE(name) \
{ \
  out << "\tExpected " << #name << ": " << expected_##name[point] << "\n"; \
  out << "\tActual " << #name << ": " << actual_##name[point] << "\n"; \
  const double rel_err = std::fabs((expected_##name[point] - actual_##name[point]) / actual_##name[point]); \
  out << "\tRelative " << #name << " error: " << rel_err << "\n"; \
  ASSERT_LT(rel_err, tolerance); \
}

  for(unsigned int point=0; point<num_points; ++point){

    out << "Point "<<point<<":\n";
    CHECK_POINT_VALUE(ux)
    CHECK_POINT_VALUE(uy)
    CHECK_POINT_VALUE(uz)
    CHECK_POINT_VALUE(sigma)
    CHECK_POINT_VALUE(tau)

    out << "\tChecking G:\n";

    out << "\t\tExpected G: ";
    for(unsigned int i=0; i<num_total_moments; ++i)
      out << std::setw(14) << expected_G[point*num_total_moments+i] << " ";
    out << "\n";

    out << "\t\tActual G:   ";
    for(unsigned int i=0; i<num_total_moments; ++i)
      out << std::setw(14) << actual_G[point*num_total_moments+i] << " ";
    out << "\n";

    double error = 0, abs_sum = 0;
    for(unsigned int i=0; i<num_ml_coefficients; ++i){
      error += std::fabs(expected_G[point*num_total_moments + i] - actual_G[point*num_total_moments + i]);
      abs_sum += std::fabs(actual_G[point*num_total_moments + i]);
    }
    out << "\tCollision Respeonse Relative Error: " << error / abs_sum << "\n";
    ASSERT_LT(error / abs_sum, 1.e-2);

    out << "\n====================================\n\n";
  }

}
