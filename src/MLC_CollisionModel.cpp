#include "MLC_CollisionModel.hpp"

// Internal Includes
#include "MLC_TensorFlowModel.hpp"
#include "MLC_Exceptions.hpp"
#include "MLC_MomentConversion.hpp"

// STL includes
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>

const std::string INPUT_DENSITY = "n";
const std::string INPUT_TEMPERATURE = "T";
const std::string INPUT_HERMITE = "F";

// TensorFlow is not a fan of giving outputs a good name - I think they are alphabetical though...
const std::string OUTPUT_TAU = "StatefulPartitionedCall:1";
const std::string OUTPUT_HERMITE = "StatefulPartitionedCall:0";

namespace mlc {

namespace {

template<typename T>
T
parseToken(const std::string & token)
{
  if(typeid(T)==typeid(double)){
    return std::stod(token);
  } else if(typeid(T) == typeid(unsigned int)){
    return static_cast<unsigned int>(std::stoi(token));
  } else if(typeid(T) == typeid(int)){
    return std::stoi(token);
  } else {
    MLC_THROW("parseToken: Not setup for type "<<typeid(T).name());
  }
}

template<typename T>
std::vector<T>
parseCommaSeparatedValues(const std::string & str)
{
  std::vector<T> vec;
  std::stringstream ss(str);
  std::string token;
  while(std::getline(ss,token,','))
    vec.push_back(parseToken<T>(token));
  return vec;
}

template<typename T>
std::vector<T>
getVectorContents(const std::string & filename,
                  const std::string & key)
{
  std::ifstream file(filename.c_str());
  std::string line;
  std::vector<T> vec;
  bool found = false;
  while(std::getline(file,line)){
    if(line.rfind(key,0) == 0){
      size_t start = line.find_first_of('[')+1;
      size_t end = line.find_first_of(']');
      found = true;
      vec = parseCommaSeparatedValues<T>(line.substr(start,end-start));
      break;
    }
  }
  file.close();

  MLC_ASSERT(found,
             "getContents : Key '"<<key<<"' not found in file '"<<filename<<"'.");

  return vec;

}


bool
isEntry(const std::string & filename,
        const std::string & key)
{
  std::ifstream file(filename.c_str());
  std::string line;
  bool found = false;
  while(std::getline(file,line)){
    if(line.rfind(key,0) == 0){
      found = true;
      break;
    }
  }

  file.close();

  return found;
}

template<typename T>
T
getScalarContents(const std::string & filename,
                  const std::string & key)
{
  std::ifstream file(filename.c_str());
  std::string line;
  T value = 0;
  bool found = false;
  while(std::getline(file,line)){
    if(line.rfind(key,0) == 0){
      size_t start = line.find_first_of(':')+1;
      value = parseToken<T>(line.substr(start,std::string::npos));
      found = true;
      break;
    }
  }
  file.close();

  MLC_ASSERT(found,
             "getContents : Key '"<<key<<"' not found in file '"<<filename<<"'.");

  return value;

}

template<class T>
constexpr const T& clamp( const T& v, const T& lo, const T& hi)
{
//  // DEBUG:
//  if(v<lo){
//    MLC_THROW("DEBUG ERROR: clamp (" << lo << ", " << v << ", " << hi << "): Value is below lower bound.");
//  } else if(v>hi){
//    MLC_THROW("DEBUG ERROR: clamp (" << lo << ", " << v << ", " << hi << "): Value is above upper bound.");
//  }

  return (v < lo) ? lo : (hi < v) ? hi : v;
}

unsigned int
convertNumCoefficientsToNumMoments(const unsigned int num_coefficients)
{
  const unsigned int num_max_moments = 20;
  for(unsigned int num_moments=3; num_moments<num_max_moments; ++num_moments){
    const unsigned int num_moment_coefficients = (num_moments+2) * (num_moments+1) * (num_moments) / 6 - 5;
    if(num_moment_coefficients == num_coefficients)
      return num_moments;
    MLC_ASSERT(num_moment_coefficients < num_coefficients,
               "convertNumCoefficientsToNumMoments : Invalid number of coefficients "<<num_coefficients <<" - overshot with "<<num_moment_coefficients<<" for "<<num_moments<<" moments.";)
  }
  MLC_THROW("convertNumCoefficientsToNumMoments : Invalid number of coefficients "<<num_coefficients <<".")
}

void
evaluatePrimitiveState(const MomentIndexer & mi,
                       const Array & M_dataset,
                       const double mass,
                       const double kB,
                       Array & n_dataset,
                       Array & ux_dataset,
                       Array & uy_dataset,
                       Array & uz_dataset,
                       Array & sigma_dataset)
{

  const unsigned int num_points = M_dataset.extent(0);
  const unsigned int num_moments = mi.getNumMoments();
  const unsigned int num_coefficients = mi.getNumCoefficients();

  if(M_dataset.extent(0)     != num_points) throw std::logic_error("CollisionModel::generateCollisionalResponse : Incorrect input allocation (M).");

  if(n_dataset.extent(0)     != num_points) throw std::logic_error("CollisionModel::generateCollisionalResponse : Incorrect input allocation (n).");
  if(ux_dataset.extent(0)    != num_points) throw std::logic_error("CollisionModel::generateCollisionalResponse : Incorrect input allocation (ux).");
  if(uy_dataset.extent(0)    != num_points) throw std::logic_error("CollisionModel::generateCollisionalResponse : Incorrect input allocation (uy).");
  if(uz_dataset.extent(0)    != num_points) throw std::logic_error("CollisionModel::generateCollisionalResponse : Incorrect input allocation (uz).");
  if(sigma_dataset.extent(0) != num_points) throw std::logic_error("CollisionModel::generateCollisionalResponse : Incorrect input allocation (sigma).");

  // 1,2) Fill g0, ux, uy, uz, sigma

  const unsigned int idx_rho = mi(0,0,0);
  const unsigned int idx_px  = mi(1,0,0);
  const unsigned int idx_py  = mi(0,1,0);
  const unsigned int idx_pz  = mi(0,0,1);
  const unsigned int idx_exx = mi(2,0,0);
  const unsigned int idx_eyy = mi(0,2,0);
  const unsigned int idx_ezz = mi(0,0,2);

  const double * const __restrict__ M_array = M_dataset.data();
  double * const __restrict__ ux_array = ux_dataset.data();
  double * const __restrict__ uy_array = uy_dataset.data();
  double * const __restrict__ uz_array = uz_dataset.data();
  double * const __restrict__ sigma_array = sigma_dataset.data();
  double * const __restrict__ n_array = n_dataset.data();

  for(unsigned int i=0; i<num_points; ++i){

    // Conserved moments
    const double * const __restrict__ Mi = M_array + i*num_coefficients;

    // Conserved moments (input)
    const double rho = Mi[idx_rho];
    const double px  = Mi[idx_px];
    const double py  = Mi[idx_py];
    const double pz  = Mi[idx_pz];
    const double eps = 0.5 * (Mi[idx_exx] + Mi[idx_eyy] + Mi[idx_ezz]);

    // Primitive moments
    const double n   = rho / mass;
    const double KE  = 0.5 * (px*px + py*py + pz*pz) / rho;
    const double P   = (2./3.) * (eps - KE);
    const double T   = P / n / kB;

    // We have to write some of these primitive moments to arrays
    ux_array[i]     = px / rho;
    uy_array[i]     = py / rho;
    uz_array[i]     = pz / rho;
    double & sigma  = sigma_array[i] = std::sqrt(P / rho);
    n_array[i]      = n;
  }
}


void
evaluateMLInputs(const MomentIndexer & mi,
                 const MomentIndexer & ml_mi,
                 const double mass,
                 const double kB,
                 const Array & M_dataset,
                 const Array & n_dataset,
                 const Array & ux_dataset,
                 const Array & uy_dataset,
                 const Array & uz_dataset,
                 const Array & sigma_dataset,
                 MLArray & ml_n_dataset,
                 MLArray & ml_T_dataset,
                 MLArray & ml_F_dataset,
                 Array & moment_scratch_0,
                 Array & moment_scratch_1)
{

  const unsigned int num_moments = mi.getNumMoments();
  const unsigned int num_points = M_dataset.extent(0);
  const unsigned int num_coefficients = mi.getNumCoefficients();
  const unsigned int num_ml_coefficients = ml_mi.getNumCoefficients();

  if(n_dataset.extent(0)     != num_points) throw std::logic_error("evaluateMLInputs : Incorrect input allocation (n).");
  if(ux_dataset.extent(0)    != num_points) throw std::logic_error("evaluateMLInputs : Incorrect input allocation (ux).");
  if(uy_dataset.extent(0)    != num_points) throw std::logic_error("evaluateMLInputs : Incorrect input allocation (uy).");
  if(uz_dataset.extent(0)    != num_points) throw std::logic_error("evaluateMLInputs : Incorrect input allocation (uz).");
  if(sigma_dataset.extent(0) != num_points) throw std::logic_error("evaluateMLInputs : Incorrect input allocation (sigma).");

  if(ml_n_dataset.extent(0)     != num_points) throw std::logic_error("evaluateMLInputs : Incorrect output allocation (n).");
  if(ml_T_dataset.extent(0)     != num_points) throw std::logic_error("evaluateMLInputs : Incorrect output allocation (T).");
  if(ml_F_dataset.extent(0)     != num_points) throw std::logic_error("evaluateMLInputs : Incorrect output allocation (F).");


  // This is required to map between the ML I/O and the moments/coefficients
  std::vector<unsigned int> ml_to_moment_index(num_ml_coefficients);
  for(unsigned int i=0; i<num_moments; ++i)
    for(unsigned int j=0; j<num_moments-i; ++j)
      for(unsigned int k=0; k<num_moments-i-j; ++k){
        int idx = ml_mi(i,j,k);
        if(idx >= 0)
          ml_to_moment_index[idx] = mi(i,j,k);
      }

  // Use G as scratch space for generating F
  auto & F_dataset = moment_scratch_0;
  auto & pM_dataset = moment_scratch_1;

  const double * const __restrict__ n_array = n_dataset.data();
  const double * const __restrict__ ux_array = ux_dataset.data();
  const double * const __restrict__ uy_array = uy_dataset.data();
  const double * const __restrict__ uz_array = uz_dataset.data();
  const double * const __restrict__ sigma_array = sigma_dataset.data();

  float * const __restrict__ ml_n_array = ml_n_dataset.data();
  float * const __restrict__ ml_T_array = ml_T_dataset.data();

  for(unsigned int i=0; i<num_points; ++i){

    // Conserved moments
    const double * const __restrict__ Mi   = M_dataset.data()   + i*num_coefficients;

    // Scratch space for handling moments
    double * const __restrict__ pMi   = pM_dataset.data()   + i*num_coefficients;
    double * const __restrict__ Fi    = F_dataset.data()    + i*num_coefficients;
    float  * const __restrict__ ml_Fi = ml_F_dataset.data() + i*num_ml_coefficients;

    // Conserved moments (input)
    const double n = n_array[i];
    const double ux = ux_array[i];
    const double uy = uy_array[i];
    const double uz = uz_array[i];
    const double sigma = sigma_array[i];

    // Get the pressure for sigma and n
    const double P = sigma * sigma * mass * n;
    const double T = P / n / kB;

    // We have enough information now to generate the primitive moments from cm
    convertConservedMomentsToNormalizedPrimitiveMoments(mass, n, ux, uy, uz, P, mi, Mi, pMi);

    // Now we generate the hermite coefficients from the primitive form
    convertNormalizedPrimitiveMomentsToHermiteCoefficients(mass, n, P, mi, pMi, Fi);

    // Now move things into the ML arrays
    ml_n_array[i] = n;
    ml_T_array[i] = T;
    for(unsigned int j=0; j<num_ml_coefficients; ++j)
      ml_Fi[j] = Fi[ml_to_moment_index[j]];


    // DEBUG
//    std::cout << "CELL " << i << "\n";
//    std::cout << "\tn: " << n << "\n";
//    std::cout << "\tT: " << T << "\n";
//    std::cout << "\tF: ";
//    for(unsigned int j=0; j<num_coefficients; ++j)
//      std::cout << Fi[j] << " ";
//    std::cout << "\n";
  }
}

void
convertMLToOutput_Hermite(const MomentIndexer & mi,
                          const MomentIndexer & ml_mi,
                          const MLArray & ml_tau_dataset,
                          const MLArray & ml_G_dataset,
                          Array & tau_dataset,
                          Array & G_dataset)
{

  const float * const __restrict__ ml_tau_array = ml_tau_dataset.data();
  const float * const __restrict__ ml_G_array = ml_G_dataset.data();

  double * const __restrict__ tau_array = tau_dataset.data();
  double * const __restrict__ G_array = G_dataset.data();

  const unsigned int num_moments = mi.getNumMoments();
  const unsigned int num_coefficients = mi.getNumCoefficients();
  const unsigned int num_ml_coefficients = ml_mi.getNumCoefficients();

  // This is required to map between the ML I/O and the moments/coefficients
  std::vector<unsigned int> ml_to_moment_index(num_ml_coefficients);
  for(unsigned int i=0; i<num_moments; ++i)
    for(unsigned int j=0; j<num_moments-i; ++j)
      for(unsigned int k=0; k<num_moments-i-j; ++k){
        int idx = ml_mi(i,j,k);
        if(idx >= 0)
          ml_to_moment_index[idx] = mi(i,j,k);
      }

  const unsigned int num_points = ml_G_dataset.extent(0);
  const unsigned int idx_Gxx = mi(2,0,0);
  const unsigned int idx_Gyy = mi(0,2,0);
  const unsigned int idx_Gzz = mi(0,0,2);

  for(unsigned int i=0; i<num_points; ++i){

    tau_array[i] = ml_tau_array[i];

    // Convert ML hermite G to hermite G
    const float * const __restrict__ ml_Gi = ml_G_array + i*num_ml_coefficients;
    double * const __restrict__ Gi = G_array + i*num_coefficients;

    // We zero out G because almost everything not filled by ML needs to be zero for conservation
    std::fill_n(Gi,num_coefficients,0.);
    for(unsigned int j=0; j<num_ml_coefficients; ++j)
      Gi[ml_to_moment_index[j]] = ml_Gi[j];

    // There is one missing non-zero term in G having to do with the energy (required for conservation)
    Gi[idx_Gxx] = - (Gi[idx_Gyy] + Gi[idx_Gzz]);

    // DEBUG
//    std::cout << "CELL " << i << "\n";
//    std::cout << "\ttau: " << tau_array[i] << "\n";
//    std::cout << "\tG: ";
//    for(unsigned int j=0; j<num_coefficients; ++j)
//      std::cout << Gi[j] << " ";
//    std::cout << "\n";
  }
}

void
convertHermiteToPrimitive(const MomentIndexer & mi,
                          const double mass,
                          const Array & hG_dataset,
                          const Array & n_dataset,
                          const Array & sigma_dataset,
                          Array & pG_dataset)
{

  const unsigned int num_points = n_dataset.extent(0);
  const unsigned int num_coefficients = mi.getNumCoefficients();

  const double * const __restrict__ hG_array = hG_dataset.data();
  double * const __restrict__ pG_array = pG_dataset.data();

  const double * const __restrict__ n_array = n_dataset.data();
  const double * const __restrict__ sigma_array = sigma_dataset.data();

  for(unsigned int i=0; i<num_points; ++i){

    const double * const __restrict__ hGi = hG_array + i * num_coefficients;
    double * const __restrict__ pGi = pG_array + i * num_coefficients;

    // Need to recreate number density and pressure for conversion to primitive coefficients
    const double n = n_array[i];
    const double sigma = sigma_array[i];
    const double P = sigma * sigma * mass * n;

    // Convert the hermite G to the primitive G
    convertHermiteCoefficientsToPrimitiveCoefficients(mass, n, P, mi, hGi, pGi);

  }

}

bool
fileExists(const std::string & filename)
{
  std::ifstream f(filename.c_str());
  return f.good();
}

}

CollisionModel::
CollisionModel(const unsigned int num_points,
               const std::string & model_name)
{

  // There are two ways to load a model
  if(fileExists(model_name+"/parameters.yaml")){
    loadParameters(model_name+"/parameters.yaml");
    model_ = std::make_shared<TensorFlowModel>(model_name+"/model", num_points);
  } else
    throw std::logic_error("CollisionModel::CollisionModel : Failed to find model '"+model_name+"'.");

  // Check for valid inputs
  {
    const auto model_tensors = model_->getInputNames();
    const std::set<std::string> required_entries {INPUT_DENSITY, INPUT_TEMPERATURE, INPUT_HERMITE};
    for(const auto & name : required_entries){
      MLC_ASSERT(model_tensors.find(name) != model_tensors.end(),
                 "MLCollisionModel::MLCollisionModel : Required input '"<<name<<"' was not found in model. Model:\n\n"<<*model_<<"\n\n");
    }
  }

  // Check for valid outputs
  {
    const auto model_tensors = model_->getOutputNames();
    const std::set<std::string> required_entries {OUTPUT_TAU, OUTPUT_HERMITE};
    for(const auto & name : required_entries){
      MLC_ASSERT(model_tensors.find(name) != model_tensors.end(),
                 "MLCollisionModel::MLCollisionModel : Required output '"<<name<<"' was not found in model. Model:\n\n"<<*model_<<"\n\n");
    }
  }

  // Get the extents of inputs to setup the hermite projection
  const auto F_extent = model_->getInputExtent(INPUT_HERMITE);
  const auto dG_extent = model_->getOutputExtent(OUTPUT_HERMITE);

  MLC_ASSERT(F_extent.size() == 2, "MLCollisionModel::MLCollisionModel : Issue with ML model inputs.");
  MLC_ASSERT(dG_extent.size() == 2, "MLCollisionModel::MLCollisionModel : Issue with ML model outputs.");

  // This is hardcoded to support the current collision operator defined in the python generator
  const unsigned int num_input_coefficients = ml_input_moment_indexer_.getNumCoefficients();
  const unsigned int num_output_coefficients = ml_output_moment_indexer_.getNumCoefficients();

  // Make sure the array sizing is as large as expected
  MLC_ASSERT(F_extent[1] == static_cast<int>(num_input_coefficients), "MLCollisionModel::MLCollisionModel : ML model expects "<<F_extent[0]<<" input coefficients, but metadata expects "<<num_input_coefficients<<" coefficients.");
  MLC_ASSERT(dG_extent[1] == static_cast<int>(num_output_coefficients), "MLCollisionModel::MLCollisionModel : ML model expects "<<dG_extent[0]<<" output coefficients, but metadata expects "<<num_output_coefficients<<" coefficients.");

  // Allocate our scratch spaces for the given number of points
  scratch_input_moments_0_ = Array({num_points, input_moment_indexer_.getNumCoefficients()});
  scratch_input_moments_1_ = Array({num_points, input_moment_indexer_.getNumCoefficients()});
  scratch_output_moments_ = Array({num_points, output_moment_indexer_.getNumCoefficients()});

  ml_tau_dataset_ = MLArray({num_points});
  ml_T_dataset_ = MLArray({num_points});
  ml_F_dataset_ = MLArray({num_points, ml_input_moment_indexer_.getNumCoefficients()});
  ml_G_dataset_ = MLArray({num_points, ml_output_moment_indexer_.getNumCoefficients()});

}


void
CollisionModel::
loadParameters(const std::string & filename)
{

  unsigned int num_input_coefficients = 0, num_output_coefficients = 0;
  if(isEntry(filename,"num_coefficients")){
    num_input_coefficients = num_output_coefficients = getScalarContents<int>(filename, "num_coefficients");
  } else {
    num_input_coefficients = getScalarContents<int>(filename, "num_input_coefficients");
    num_output_coefficients = getScalarContents<int>(filename, "num_output_coefficients");
  }

  ml_input_moment_indexer_.setupForCoefficients(num_input_coefficients);
  input_moment_indexer_.setupForMoments(ml_input_moment_indexer_.getNumMoments());

  ml_output_moment_indexer_.setupForCoefficients(num_output_coefficients);
  output_moment_indexer_.setupForMoments(ml_output_moment_indexer_.getNumMoments());

  kB_ = getScalarContents<double>(filename,"kB");
  mass_ = getScalarContents<double>(filename,"mass");

  n_scaling_ = Scaling(getScalarContents<double>(filename,"n_min"),
                       getScalarContents<double>(filename,"n_max"));

  T_scaling_ = Scaling(getScalarContents<double>(filename,"T_min"),
                       getScalarContents<double>(filename,"T_max"));

  tau_scaling_ = Scaling(getScalarContents<double>(filename,"tau_min"),
                         getScalarContents<double>(filename,"tau_max"));

  {
    const auto F_min = getVectorContents<double>(filename,"F_min");
    const auto F_max = getVectorContents<double>(filename,"F_max");
    MLC_ASSERT(F_min.size() == num_input_coefficients,
               "MLDSMCIndexer::load : F_min is incorrect size.");
    MLC_ASSERT(F_max.size() == num_input_coefficients,
               "MLDSMCIndexer::load : F_max is incorrect size.");
    F_scaling_.resize(num_input_coefficients);
    for(unsigned int i=0; i<num_input_coefficients; ++i)
      F_scaling_[i] = Scaling(F_min[i],F_max[i]);
  }

  {
    const auto dG_min = getVectorContents<double>(filename,"dG_min");
    const auto dG_max = getVectorContents<double>(filename,"dG_max");
    MLC_ASSERT(dG_min.size() == num_output_coefficients,
               "MLDSMCIndexer::load : dG_min is incorrect size.");
    MLC_ASSERT(dG_max.size() == num_output_coefficients,
               "MLDSMCIndexer::load : dG_max is incorrect size.");
    G_scaling_.resize(num_output_coefficients);
    for(unsigned int i=0; i<num_output_coefficients; ++i)
      G_scaling_[i] = Scaling(dG_min[i],dG_max[i]);
  }

}



void
CollisionModel::
generateCollisionalResponse(const Array & M_dataset,
                            Array & tau_dataset,
                            Array & g0_dataset,
                            Array & ux_dataset,
                            Array & uy_dataset,
                            Array & uz_dataset,
                            Array & sigma_dataset,
                            Array & G_dataset,
                            const bool primitive_output) const
{

  using Scalar = float;

  /**
   *
   * The goal of this operator is to read in the conserved moments (M) and generate a set of outputs that can be used to
   * evaluate the collisional response on a spectral monomial basis.
   * We use a monomial polynomial set because that is a lot easier to work with than the hermite coefficients.
   *
   * 1) Convert conserved moments to normalized primitive moments
   *    - Fill g0, ux, uy, uz, sigma
   * 2) Convert normalized primitive moments to hermite coefficients
   * 3) Run model using hermite coefficients
   *    - Fill tau
   * 4) Convert hermite coefficients to normalized primitive moments
   *    - Fill G
   *
   */

  // Scratch space
  auto & n_dataset = g0_dataset;

  // Generate n, ux, uy, uz, sigma from M
  evaluatePrimitiveState(input_moment_indexer_, M_dataset, mass_, kB_, n_dataset, ux_dataset, uy_dataset, uz_dataset, sigma_dataset);

  // Scratch space
  auto & ml_F_dataset = ml_F_dataset_;
  auto & ml_n_dataset = ml_tau_dataset_;
  auto & ml_T_dataset = ml_T_dataset_;

  // Convert the inputs to the ML inputs
  evaluateMLInputs(input_moment_indexer_, ml_input_moment_indexer_, mass_, kB_,
                   M_dataset, n_dataset, ux_dataset, uy_dataset, uz_dataset, sigma_dataset,
                   ml_n_dataset, ml_T_dataset, ml_F_dataset,
                   scratch_input_moments_0_, scratch_input_moments_1_);

  // Evaluate model
  evaluateModel(ml_n_dataset, ml_T_dataset, ml_F_dataset, ml_tau_dataset_, ml_G_dataset_);

  if(primitive_output){

    // Convert outputs from their ML form to a Hermite form
    convertMLToOutput_Hermite(output_moment_indexer_, ml_output_moment_indexer_, ml_tau_dataset_, ml_G_dataset_,
                              tau_dataset, scratch_output_moments_);

    convertHermiteToPrimitive(output_moment_indexer_, mass_, scratch_output_moments_, n_dataset, sigma_dataset, G_dataset);

  } else {

    // Convert outputs from their ML form to a Hermite form
    convertMLToOutput_Hermite(output_moment_indexer_, ml_output_moment_indexer_, ml_tau_dataset_, ml_G_dataset_,
                              tau_dataset, G_dataset);

  }

  // Create the output Array for the scaling
  const double mult = primitive_output ? 1. / std::pow(2 * M_PI,1.5) : 1. / std::pow(2 * std::sqrt(M_PI),1.5) ;
  for(unsigned int i=0; i<M_dataset.extent(0); ++i)
    g0_dataset[i] = mult * n_dataset[i] / std::pow(sigma_dataset[i],3);

}


void
CollisionModel::
evaluateModel(MLArray & n,
              MLArray & T,
              MLArray & F,
              MLArray & tau,
              MLArray & G) const
{

  const unsigned int num_points = ml_tau_dataset_.extent(0);
  const unsigned int num_input_coefficients = F.extent(1);
  const unsigned int num_output_coefficients = ml_G_dataset_.extent(1);

  MLC_ASSERT(n.extent(0)   == num_points, "MLCollisionModel::evaluateModel : Input 'n' has wrong size.");
  MLC_ASSERT(T.extent(0)   == num_points, "MLCollisionModel::evaluateModel : Input 'T' has wrong size.");
  MLC_ASSERT(F.extent(0)   == num_points, "MLCollisionModel::evaluateModel : Input 'F' has wrong size.");
  MLC_ASSERT(F.extent(1)   == num_input_coefficients, "MLCollisionModel::evaluateModel : Input 'F' has wrong size.");

  MLC_ASSERT(tau.extent(0) == num_points, "MLCollisionModel::evaluateModel : Output 'tau' has wrong size.");
  MLC_ASSERT(G.extent(0)   == num_points, "MLCollisionModel::evaluateModel : Output 'G' has wrong size.");
  MLC_ASSERT(G.extent(1)   == num_output_coefficients, "MLCollisionModel::evaluateModel : Output 'G' has wrong size.");

  // First rescale the inputs
  for(unsigned int i=0; i<num_points; ++i){
    n[i] = n_scaling_.xToY(n[i]);
    T[i] = T_scaling_.xToY(T[i]);
    for(unsigned int j=0,idx=i*num_input_coefficients; j<num_input_coefficients; ++j,++idx)
      F[idx] = F_scaling_[j].xToY(F[idx]);
  }

  // Copy to ML inputs
  model_->setInputData(INPUT_DENSITY,n.data());
  model_->setInputData(INPUT_TEMPERATURE,T.data());
  model_->setInputData(INPUT_HERMITE,F.data());

  // Run model
  model_->run();

  // Copy from ML outputs
  model_->getOutputData(OUTPUT_TAU,tau.data());
  model_->getOutputData(OUTPUT_HERMITE,G.data());

  // Now rescale the outputs
  for(unsigned int i=0; i<num_points; ++i){
    tau[i] = tau_scaling_.yToX(tau[i]);
    for(unsigned int j=0,idx=i*num_output_coefficients; j<num_output_coefficients; ++j,++idx)
      G[idx] = G_scaling_[j].yToX(G[idx]);
  }

}

}
