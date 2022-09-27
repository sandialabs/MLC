#ifndef MLC_TensorFlowGraph_hpp_
#define MLC_TensorFlowGraph_hpp_

// STL includes
#include <string>
#include <vector>
#include <set>
#include <map>

// Internal includes
#include "MLC_Matrix.hpp"

struct TF_Session;
struct TF_Graph;
struct TF_Operation;
struct TF_Output;

namespace mlc {

/// Used to store TF metadata for interacting with the graph
struct Node
{
  TF_Operation * op;
  std::vector<int> input_indexes;
  std::vector<int> output_indexes;
};

class
TensorFlowGraph
{
public:

  /// Block default constructor
  TensorFlowGraph() = delete;

  /// Construct tensorflow model given a directory containing a saved model
  TensorFlowGraph(const std::string & model_directory);

  /// Default destructor
  ~TensorFlowGraph();

  /// Get the names of all the nodes in the graph that have useful inputs and outputs
  std::set<std::string>
  getNodeNames() const;

  /// Check if an node/operation exists
  bool
  isNode(const std::string & name) const;

  /// Get the number of outputs for a given node/operations - only lists inputs with float, double, int, or uint types
  int
  getNumNodeInputs(const std::string & name) const;

  /// Get the number of outputs for a given node/operations - only lists inputs with float, double, int, or uint types
  int
  getNumNodeOutputs(const std::string & name) const;

  /// Get the extent for a node/operation input
  std::vector<int>
  getNodeOutputExtent(const std::string & name,
                      const int index) const;

  /// Get the extent for a node/operation output
  std::vector<int>
  getNodeInputExtent(const std::string & name,
                     const int index) const;

  /// Get the datatype for a given input to a node
  const std::type_info &
  getNodeInputDataType(const std::string & name,
                       const int index) const;

  /// Get the datatype for a given output from a node
  const std::type_info &
  getNodeOutputDataType(const std::string & name,
                        const int index) const;

  /// Get the name of the node that outputs the given input
  std::string
  getNodeInputSupplier(const std::string & name,
                       const int index) const;

  /// When all else fails, we give direct access to the internal nodes - NEED TO DEPRECATE AND FIGURE OUT A BETTER DESIGN
  const Node &
  getNode(const std::string & name) const;

  /// Get the weights/kernel for a given layer
  template<typename T>
  TMatrix<T>
  getLayerWeights(const std::string & name) const;

  /// Get the bias/offsets for the given layer
  template<typename T>
  TVector<T>
  getLayerBias(const std::string & name) const;

  /// Get the underlying tensorflow graph
  TF_Graph *
  getGraph() const;

  /// Get the underlying tensorflow session
  TF_Session *
  getSession() const;

  /// This writes out all the internal TensorFlow metadata for the stored model (different from the stream operator<<)
  void
  toStream(std::ostream & os) const;

protected:

  /// C API Session
  TF_Session * session_;

  /// C API Graph
  TF_Graph * graph_;

  /// Names of nodes
  std::map<std::string, Node> nodes_;

};

}

std::ostream &
operator<<(std::ostream & os,
           const mlc::TensorFlowGraph & model);

#endif
