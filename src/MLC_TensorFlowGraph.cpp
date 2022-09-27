#include "MLC_TensorFlowGraph.hpp"

// STL includes
#include <iostream>

// Internal includes
#include "MLC_Exceptions.hpp"
#include "MLC_TFUtilities.hpp"
#include "MLC_Utilities.hpp"

namespace mlc {

namespace {

bool
isValidTensorType(TF_Graph * graph,
                  const TF_Output & field)
{
  auto type = TF_OperationOutputType(field);

  // We severely restrict what we are allowed to work with
  if((type == TF_FLOAT) or (type == TF_DOUBLE) or (type == TF_INT32) or (type == TF_UINT32))
    return getDims(graph,field).size() > 0;

  return false;
}

}

TensorFlowGraph::
TensorFlowGraph(const std::string & model_directory):
  session_(nullptr),
  graph_(TF_NewGraph())
{

  TF_Status * status = TF_NewStatus();

  // Build the session
  {

    // Some required metadata
    TF_SessionOptions * session_options = TF_NewSessionOptions();
    const char * tag = "serve";

    // Not sure how to set these
    TF_Buffer * run_options = nullptr;
    TF_Buffer * meta_graph_def = nullptr;

    // Load the session
    session_ = TF_LoadSessionFromSavedModel(session_options, run_options,
                                            model_directory.c_str(),
                                            &tag,
                                            1,
                                            graph_,meta_graph_def,status);

    // Make sure the session was loaded properly
    if(TF_GetCode(status) != TF_OK){
      const std::string status_msg = TF_Message(status);
      TF_DeleteSessionOptions(session_options);
      TF_DeleteStatus(status);
      status = nullptr;
      MLC_THROW("TensorFlowSession::TensorFlowSession : Failed to create session. Message:\n\n"<<TF_Message(status)<<"\n\n")
    }

    // Cleanup
    TF_DeleteSessionOptions(session_options);

  }

  // Grab the names of all the nodes
  {
    std::size_t pos = 0;
    TF_Operation * op = nullptr;
    while( (op = TF_GraphNextOperation(graph_, &pos)) != nullptr){

      const std::string name = TF_OperationName(op);
      Node node;
      node.op = op;

      int num_total_inputs = TF_OperationNumInputs(op);
      int num_total_outputs = TF_OperationNumOutputs(op);

      for(int i=0; i<num_total_inputs; ++i){
        TF_Input input {op,i};
        auto type = TF_OperationInputType(input);
        // We have to find the supplier for this input to determine the extent of the tensor
        auto output = TF_OperationInput(input);
        if(isValidTensorType(graph_, output))
          node.input_indexes.push_back(input.index);
      }
      for(int i=0; i<num_total_outputs; ++i){
        TF_Output output {op,i};
        auto type = TF_OperationOutputType(output);
        if(isValidTensorType(graph_,output))
          node.output_indexes.push_back(output.index);
      }
      if(node.input_indexes.size() + node.output_indexes.size() > 0)
        nodes_[name] = node;
    }
  }

  TF_DeleteStatus(status);

}

TensorFlowGraph::
~TensorFlowGraph()
{
  if(graph_ != nullptr){
    TF_DeleteGraph(graph_);
    graph_ = nullptr;
  }
  if(session_ != nullptr){
    TF_Status * status = TF_NewStatus();
    TF_CloseSession(session_,status);
    if(TF_GetCode(status) != TF_OK){
      std::string status_msg = TF_Message(status);
      TF_DeleteStatus(status);
      std::cerr << "TensorFlowGraph::~TensorFlowGraph : Failed to delete session. Message:\n\n"<<TF_Message(status)<<"\n\n";
    } else
      session_ = nullptr;
    TF_DeleteStatus(status);
  }
}

std::set<std::string>
TensorFlowGraph::
getNodeNames() const
{
  std::set<std::string> names;
  for(const auto & pr : nodes_)
    names.insert(pr.first);
  return names;
}

bool
TensorFlowGraph::
isNode(const std::string & name) const
{
  return nodes_.find(name) != nodes_.end();
}

int
TensorFlowGraph::
getNumNodeInputs(const std::string & name) const
{
  const auto itr = nodes_.find(name);
  MLC_ASSERT(itr != nodes_.end(),
             "TensorFlowGraph::getNumNodeInputs : Node '"<<name<<"' does not exist in graph.");
  return itr->second.input_indexes.size();
}

int
TensorFlowGraph::
getNumNodeOutputs(const std::string & name) const
{
  const auto itr = nodes_.find(name);
  MLC_ASSERT(itr != nodes_.end(),
             "TensorFlowGraph::getNumNodeOutputs : Node '"<<name<<"' does not exist in graph.");
  return itr->second.output_indexes.size();
}

std::vector<int>
TensorFlowGraph::
getNodeInputExtent(const std::string & name,
                   const int index) const
{
  const auto itr = nodes_.find(name);
  MLC_ASSERT(itr != nodes_.end(),
             "TensorFlowGraph::getNodeInputExtent : Node '"<<name<<"' does not exist in graph.");
  const auto & node = itr->second;
  MLC_ASSERT((index >= 0) and (index < int(node.input_indexes.size())),
             "TensorFlowGraph::getNodeInputExtent : Index out of range.");

  // Oddly enough, inputs don't have extents associated with them, you have to find the node that outputs this input
  TF_Input input{node.op, node.input_indexes[index]};
  auto output = TF_OperationInput(input);
  return getDims(graph_, output);
}

std::vector<int>
TensorFlowGraph::
getNodeOutputExtent(const std::string & name,
                    const int index) const
{
  const auto itr = nodes_.find(name);
  MLC_ASSERT(itr != nodes_.end(),
             "TensorFlowGraph::getNodeOutputExtent : Node '"<<name<<"' does not exist in graph.");
  const auto & node = itr->second;
  MLC_ASSERT((index >= 0) and (index < int(node.output_indexes.size())),
            "TensorFlowGraph::getNodeOutputExtent : Index out of range.");
  TF_Output output{node.op, node.output_indexes[index]};
  return getDims(graph_, output);
}


const std::type_info &
TensorFlowGraph::
getNodeInputDataType(const std::string & name,
                     const int index) const
{
  const auto itr = nodes_.find(name);
  MLC_ASSERT(itr != nodes_.end(),
             "TensorFlowGraph::getNodeInputDataType : Node '"<<name<<"' does not exist in graph.");
  const auto & node = itr->second;
  MLC_ASSERT((index >= 0) and (index < int(node.input_indexes.size())),
             "TensorFlowGraph::getNodeInputDataType : Index out of range.");
  TF_Input input{node.op, node.input_indexes[index]};
  return getTypeInfo(TF_OperationInputType(input));
}

const std::type_info &
TensorFlowGraph::
getNodeOutputDataType(const std::string & name,
                      const int index) const
{
  const auto itr = nodes_.find(name);
  MLC_ASSERT(itr != nodes_.end(),
             "TensorFlowGraph::getNodeOutputDataType : Node '"<<name<<"' does not exist in graph.");
  const auto & node = itr->second;
  MLC_ASSERT((index >= 0) and (index < int(node.output_indexes.size())),
             "TensorFlowGraph::getNodeOutputDataType : Index out of range.");
  TF_Output output{node.op, node.output_indexes[index]};
  return getTypeInfo(TF_OperationOutputType(output));
}

std::string
TensorFlowGraph::
getNodeInputSupplier(const std::string & name,
                     const int index) const
{
  const auto itr = nodes_.find(name);
  MLC_ASSERT(itr != nodes_.end(),
             "TensorFlowGraph::getNodeInputDataType : Node '"<<name<<"' does not exist in graph.");
  const auto & node = itr->second;
  MLC_ASSERT((index >= 0) and (index < int(node.input_indexes.size())),
             "TensorFlowGraph::getNodeInputDataType : Index out of range.");
  TF_Input input{node.op, node.input_indexes[index]};
  TF_Output output = TF_OperationInput(input);
  return TF_OperationName(output.oper);
}

const Node &
TensorFlowGraph::
getNode(const std::string & name) const
{
  const auto itr = nodes_.find(name);
  MLC_ASSERT(itr != nodes_.end(),
             "TensorFlowGraph::getNode : Node '"<<name<<"' does not exist in graph.");
  return itr->second;
}

template<typename T>
TMatrix<T>
TensorFlowGraph::
getLayerWeights(const std::string & name) const
{
  // Layers for weights are called kernels and can only be accessed through a read op
  const std::string weights_layer = name+"/kernel/Read/ReadVariableOp";

  // Make sure the layer exists
  const auto itr = nodes_.find(weights_layer);
  MLC_ASSERT(itr != nodes_.end(),
             "TensorFlowGraph::getLayerWeights : Layer '"<<name<<"' not found in graph.");
  const auto & node = itr->second;
  MLC_ASSERT(node.output_indexes.size() == 1,
             "TensorFlowGraph::getLayerWeights : Layer '"<<name<<"' does not appear to be correctly formatted.");

  TF_Output weights_output {node.op, node.output_indexes[0]};

  MLC_ASSERT(getTypeInfo(TF_OperationOutputType(weights_output)) == typeid(T),
             "TensorFlowGraph::getLayerWeights : Weights matrix is of type "<< demangleTypeInfoName(getTypeInfo(TF_OperationOutputType(weights_output))) <<" while requested type is "<<demangleTypeInfoName(typeid(T))<<".");

  auto extent = getDims(graph_,weights_output);

  MLC_ASSERT(extent.size() == 2,
             "TensorFlowGraph::getLayerWeights : Rank of weights matrix is " << extent.size() << ", expected 2.");

  TMatrix<T> weights(extent[0], extent[1], 0.);

  const int size_in_bytes = weights.size() * sizeof(T);

  int64_t dims[2] = {extent[0], extent[1]};

  TF_Tensor * weights_tensor = TF_AllocateTensor(getDataType(typeid(T)), dims, 2, size_in_bytes);

  // Sanity check - not entirely sure why we have to pass in size_in_bytes to the above array
  MLC_ASSERT(TF_TensorByteSize(weights_tensor) == size_in_bytes,
             "TensorFlowGraph::getLayerWeights : Size mismatch between tensor and matrix.");

  TF_Status * status = TF_NewStatus();

  TF_SessionRun(session_, nullptr,
                nullptr, nullptr, 0,
                &weights_output, &weights_tensor, 1,
                &weights_output.oper, 1,
                nullptr, status);

  if(TF_GetCode(status) != TF_OK){
    const std::string status_msg = TF_Message(status);
    TF_DeleteStatus(status);
    MLC_THROW("TensorFlowGraph::getLayerWeights : Failed to run model. Message:\n\n"<<TF_Message(status)<<"\n\n")
  }
  TF_DeleteStatus(status);

  /// Copy memory from tensor to weights matrix
  memcpy(static_cast<void*>(weights.data()), TF_TensorData(weights_tensor), size_in_bytes);

  // Delete the local tensor allocation
  TF_DeleteTensor(weights_tensor);

  return weights;
}

template<typename T>
TVector<T>
TensorFlowGraph::
getLayerBias(const std::string & name) const
{

  // Layers for weights are called kernels and can only be accessed through a read op
  const std::string weights_layer = name+"/bias/Read/ReadVariableOp";

  // Make sure the layer exists
  const auto itr = nodes_.find(weights_layer);
  MLC_ASSERT(itr != nodes_.end(),
             "TensorFlowGraph::getLayerBias : Layer '"<<name<<"' not found in graph.");
  const auto & node = itr->second;
  MLC_ASSERT(node.output_indexes.size() == 1,
             "TensorFlowGraph::getLayerBias : Layer '"<<name<<"' does not appear to be correctly formatted.");

  TF_Output weights_output {node.op, node.output_indexes[0]};

  MLC_ASSERT(getTypeInfo(TF_OperationOutputType(weights_output)) == typeid(T),
             "TensorFlowGraph::getLayerBias : Weights matrix is of type "<< demangleTypeInfoName(getTypeInfo(TF_OperationOutputType(weights_output))) <<" while requested type is "<<demangleTypeInfoName(typeid(T))<<".");

  auto extent = getDims(graph_,weights_output);

  MLC_ASSERT(extent.size() == 1,
             "TensorFlowGraph::getLayerBias : Rank of weights matrix is " << extent.size() << ", expected 1.");

  TVector<T> bias(extent[0]);

  const int size_in_bytes = bias.size() * sizeof(T);

  int64_t dims[1] = {extent[0]};

  TF_Tensor * bias_tensor = TF_AllocateTensor(getDataType(typeid(T)), dims, 1, size_in_bytes);

  // Sanity check - not entirely sure why we have to pass in size_in_bytes to the above array
  MLC_ASSERT(TF_TensorByteSize(bias_tensor) == size_in_bytes,
             "TensorFlowGraph::getLayerBias : Size mismatch between tensor and matrix.");

  TF_Status * status = TF_NewStatus();

  TF_SessionRun(session_, nullptr,
                nullptr, nullptr, 0,
                &weights_output, &bias_tensor, 1,
                &weights_output.oper, 1,
                nullptr, status);

  if(TF_GetCode(status) != TF_OK){
    const std::string status_msg = TF_Message(status);
    TF_DeleteStatus(status);
    MLC_THROW("TensorFlowGraph::getLayerBias : Failed to run model. Message:\n\n"<<TF_Message(status)<<"\n\n")
  }
  TF_DeleteStatus(status);

  /// Copy memory from tensor to weights matrix
  memcpy(static_cast<void*>(bias.data()), TF_TensorData(bias_tensor), size_in_bytes);

  // Delete the local tensor allocation
  TF_DeleteTensor(bias_tensor);

  return bias;
}

TF_Graph *
TensorFlowGraph::
getGraph() const
{
  return graph_;
}

TF_Session *
TensorFlowGraph::
getSession() const
{
  return session_;
}

void
TensorFlowGraph::
toStream(std::ostream & os) const
{
  mlc::printGraph(os,graph_);
}

}

std::ostream &
operator<<(std::ostream & os,
           const mlc::TensorFlowGraph & graph)
{
  os << "Nodes:\n";
  for(const auto & name : graph.getNodeNames()){
    os << "  " << name << ":\n";
    const int num_inputs = graph.getNumNodeInputs(name);
    if(num_inputs > 0){
      os << "    Inputs:\n";
      for(int i=0; i<num_inputs; ++i)
        os << "      " << i << ": " << demangleTypeInfoName(graph.getNodeInputDataType(name,i)) << " " << toString(graph.getNodeInputExtent(name,i)) << " - from " << graph.getNodeInputSupplier(name,i)<<"\n";
    }
    const int num_outputs = graph.getNumNodeOutputs(name);
    if(num_outputs > 0){
      os << "    Outputs:\n";
      for(int i=0; i<num_outputs; ++i)
        os << "      " << i << ": " << demangleTypeInfoName(graph.getNodeOutputDataType(name,i)) << " " << toString(graph.getNodeOutputExtent(name,i)) << "\n";
    }
  }
  return os;
}


#define ETI_SETGET(type) \
template mlc::TMatrix<type> mlc::TensorFlowGraph::getLayerWeights<type>(const std::string &) const; \
template mlc::TVector<type> mlc::TensorFlowGraph::getLayerBias<type>(const std::string &) const;

ETI_SETGET(float)
ETI_SETGET(double)
ETI_SETGET(int)
ETI_SETGET(unsigned int)
