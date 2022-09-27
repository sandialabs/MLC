#ifndef MLC_TensorFlowModel_hpp_
#define MLC_TensorFlowModel_hpp_

// STL includes
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <map>

struct TF_Operation;

namespace mlc {

/// This is an internal object - user has no access
struct TensorFlowTensor;
class TensorFlowGraph;

class
TensorFlowModel
{
public:

  /// Block default constructor
  TensorFlowModel() = delete;

  /// Construct tensorflow model given a directory containing a saved model
  TensorFlowModel(const std::string & model_directory,
                  const unsigned int num_points = 1);

  /// Default destructor
  ~TensorFlowModel() = default;

  /// Get a list of input names
  std::set<std::string>
  getInputNames() const;

  /// Get a list of output names
  std::set<std::string>
  getOutputNames() const;

  /**
   * \brief Get the extent for an input with a given name
   *
   * \throws if name is not in getInputNames
   */
  const std::vector<int> &
  getInputExtent(const std::string & name) const;

  /**
   * \brief Get the extent for an output with a given name
   *
   * \throws if name is not in getOutputNames
   */
  const std::vector<int> &
  getOutputExtent(const std::string & name) const;

  /**
   * \brief Run the tensorflow model
   *
   * \note This should only be called after setInputData has been called for all inputs
   */
  void
  run();

  /// Get the datatype for the given input
  const std::type_info &
  getInputType(const std::string & name) const;

  /// Get the datatype for the given output
  const std::type_info &
  getOutputType(const std::string & name) const;

  /// Set the data for a given input
  template<typename T>
  void
  setInputData(const std::string & name,
               const T * src);

  /// Use after run is called to get the data generated by the model
  template<typename T>
  void
  getOutputData(const std::string & name,
                T * dest) const;

  /// Get the graph representing the tensorflow components
  const TensorFlowGraph &
  getGraph() const;

protected:

  /// Get the input tensor with the given name
  const TensorFlowTensor &
  getInputTensor(const std::string & name) const;

  /// Get the output tensor with the given name
  const TensorFlowTensor &
  getOutputTensor(const std::string & name) const;

  /// C API Session
  std::shared_ptr<const TensorFlowGraph> graph_;

  /// C API Model operation
  TF_Operation * model_op_;

  /// Map of names to inputs
  std::vector<std::shared_ptr<TensorFlowTensor>> inputs_;

  /// Map of names to outputs
  std::vector<std::shared_ptr<TensorFlowTensor>> outputs_;

};

}

std::ostream &
operator<<(std::ostream & os,
           const mlc::TensorFlowModel & model);

#endif
