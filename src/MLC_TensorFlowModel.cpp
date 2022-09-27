#include "MLC_TensorFlowModel.hpp"

// STL includes
#include <iostream>

// Internal includes
#include "MLC_Exceptions.hpp"
#include "MLC_TFUtilities.hpp"
#include "MLC_Utilities.hpp"
#include "MLC_TensorFlowGraph.hpp"

namespace mlc {

struct
TensorFlowTensor
{
  TensorFlowTensor() = delete;

  TensorFlowTensor(const std::string & name_,
                   TF_Graph * graph,
                   const TF_Output & output,
                   const int base_dim):
    name_(name_),
    field_(output)
  {

    extent_ = getDims(graph, field_);
    MLC_ASSERT(extent_.size() > 1,
               "TensorFlowTensor::TensorFlowTensor : Currently all tensors must be greater than 1D.");
    MLC_ASSERT(extent_[0] == -1,
               "TensorFlowTensor::TensorFlowTensor : Attempted to initialize tensor with base index != -1 (implies it has an arbitrary dimension)");
    extent_[0] = base_dim;

    std::vector<int64_t> dims;
    for(const auto d : extent_)
      dims.push_back(d);

    // Allocate the tensor
    tensor_ = TF_AllocateTensor(getDataType(),dims.data(),dims.size(), getSizeInBytes());

    // Zero the tensor
    memset(TF_TensorData(tensor_), 0, getSizeInBytes());

  }

  ~TensorFlowTensor()
  {
    // We only own the tensor - everything else is owned by someone else
    TF_DeleteTensor(tensor_);
  }

  const std::string &
  getName() const
  { return name_; }

  TF_DataType
  getDataType() const
  { return TF_OperationOutputType(field_); }

  const std::type_info &
  getType() const
  { return getTypeInfo(getDataType()); }

  const TF_Output &
  getField() const
  {return field_;}

  TF_Tensor *
  getTensor() const
  { return tensor_; }

  template<typename T>
  void
  copyIntoTensor(const T * array) const
  { memcpy(TF_TensorData(tensor_), static_cast<const void*>(array), getSizeInBytes()); }

  template<typename T>
  void
  copyFromTensor(T * array) const
  { memcpy(static_cast<void*>(array), TF_TensorData(tensor_), getSizeInBytes()); }

  const std::vector<int> &
  getExtent() const
  {return extent_;}

  size_t
  getSize() const
  {
    size_t size = 1;
    for(const auto d : extent_) size *= d;
    return size;
  }

  size_t
  getSizeInBytes() const
  { return getSize() * TF_DataTypeSize(getDataType()); }

  void
  setTensor(TF_Tensor * tensor)
  {

    // Check data type
    MLC_ASSERT(TF_TensorType(tensor) == TF_TensorType(tensor_),
               "TensorFlowTensor::setTensor : Attempted to set tensor with different type.");

    // Check tensor size
    MLC_ASSERT(TF_TensorByteSize(tensor) == TF_TensorByteSize(tensor_),
               "TensorFlowTensor::setTensor : Attempted to set tensor with different size");

    // Delete existing tensor
    TF_DeleteTensor(tensor_);

    // Swap the tensor
    tensor_ = tensor;
  }

protected:

  std::string name_;
  TF_Output field_;
  std::vector<int> extent_;
  TF_Tensor * tensor_;

};

TensorFlowModel::
TensorFlowModel(const std::string & model_directory,
                const unsigned int num_points):
  graph_(new TensorFlowGraph(model_directory))
{

  // Tensorflow has a real issue with naming...

  // Grab the operation that defines the model we wish to infer
  const std::string model_name = "StatefulPartitionedCall";
  MLC_ASSERT(graph_->isNode(model_name),
             "TensorFlowModel::TensorFlowModel : Model/operation '"<<model_name<<"' not found in graph.")

  model_op_ = TF_GraphOperationByName(graph_->getGraph(),model_name.c_str());
  MLC_ASSERT(model_op_ != nullptr,
             "TensorFlowModel::TensorFlowModel : Model/operation '"<<model_name<<"' found in graph, but cannot be accessed.")

  // Grab all inputs to the model
  const auto & node = graph_->getNode(model_name);
  for(int i=0; i<int(node.input_indexes.size()); ++i){

    TF_Input input{node.op, node.input_indexes[i]};
    TF_Output supplier = TF_OperationInput(input);

    // Get the name of the node that supplies this input
    const std::string supplier_name = TF_OperationName(supplier.oper);

    // Right now we can only identify inputs to the models by scanning all operators that start with "serving_default_" and finding their 1 output (will have no inputs)
    const std::string prefix = "serving_default_";
    MLC_ASSERT(supplier_name.find(prefix) == 0,
               "TensorFlowModel::TensorFlowModel : Unexpected input supplier name '"<<supplier_name<<"' - we currently assume those start with '"<<prefix<<"'")

    const std::string input_name = supplier_name.substr(prefix.size());

    MLC_ASSERT(graph_->getNumNodeInputs(supplier_name) == 0,
               "TensorFlowModel::TensorFlowModel : Found input operation '"<<supplier_name<<"' with '"<<graph_->getNumNodeInputs(supplier_name)<<"' input fields (expected 0).");

    MLC_ASSERT(graph_->getNumNodeOutputs(supplier_name) == 1,
               "TensorFlowModel::TensorFlowModel : Found input operation '"<<supplier_name<<"' with '"<<graph_->getNumNodeOutputs(supplier_name)<<"' output fields (expected 1).");

    for(const auto & input : inputs_){
      MLC_ASSERT(input->getName() != input_name,
                 "TensorFlowModel::TensorFlowModel : Found input '"<<input_name<<"', which already exists.");
    }

    // Setting tensor data to the input requires a TF_Output object
    auto tensor = std::make_shared<TensorFlowTensor>(input_name, graph_->getGraph(), supplier, num_points);
    inputs_.push_back(tensor);
  }

  // Grab all outputs to the model
  {
    // Outputs are defined by the model op
    for(int i=0; i<int(node.output_indexes.size()); ++i){
      const int index = node.output_indexes[i];
      // TODO: Figure out how to read the output name from tensorflow (for now we will double down on bad names)
      TF_Output output{node.op,index};
      auto tensor = std::make_shared<TensorFlowTensor>(model_name+":" + std::to_string(index), graph_->getGraph(), output, num_points);
      outputs_.push_back(tensor);
    }
  }

}

std::set<std::string>
TensorFlowModel::
getInputNames() const
{
  std::set<std::string> names;
  for(const auto & var : inputs_)
    names.insert(var->getName());
  return names;
}

std::set<std::string>
TensorFlowModel::
getOutputNames() const
{
  std::set<std::string> names;
  for(const auto & var : outputs_)
    names.insert(var->getName());
  return names;
}

const std::vector<int> &
TensorFlowModel::
getInputExtent(const std::string & name) const
{
  return getInputTensor(name).getExtent();
}

const std::vector<int> &
TensorFlowModel::
getOutputExtent(const std::string & name) const
{
  return getOutputTensor(name).getExtent();
}

const TensorFlowTensor &
TensorFlowModel::
getInputTensor(const std::string & name) const
{
  for(const auto & var : inputs_)
    if(var->getName() == name)
      return *var;
  MLC_THROW("TensorFlowModel::getInputTensor : Could not find input '"<<name<<"'");
}

const TensorFlowTensor &
TensorFlowModel::
getOutputTensor(const std::string & name) const
{
  for(const auto & var : outputs_)
    if(var->getName() == name)
      return *var;
  MLC_THROW("TensorFlowModel::getOutputTensor : Could not find output '"<<name<<"'");
}

void
TensorFlowModel::
run()
{

  std::vector<TF_Output> inputs(inputs_.size());
  std::vector<TF_Tensor*> input_tensors(inputs_.size(),nullptr);

  std::vector<TF_Output> outputs(outputs_.size());
  std::vector<TF_Tensor*> output_tensors(outputs_.size(),nullptr);

  for(unsigned int i=0; i<inputs_.size(); ++i){
    inputs[i] = inputs_[i]->getField();
    input_tensors[i] = inputs_[i]->getTensor();
  }
  for(unsigned int i=0; i<outputs_.size(); ++i)
    outputs[i] = outputs_[i]->getField();

  TF_Buffer* run_options = nullptr;
  TF_Buffer* run_metadata = nullptr;
  TF_Status * status = TF_NewStatus();

  TF_SessionRun(graph_->getSession(), run_options,
                inputs.data(), input_tensors.data(), input_tensors.size(),
                outputs.data(), output_tensors.data(), output_tensors.size(),
                &model_op_, 1,
//                nullptr, 0,
                run_metadata, status);

  if(TF_GetCode(status) != TF_OK){
    const std::string status_msg = TF_Message(status);
    TF_DeleteStatus(status);
    MLC_THROW("TensorFlowModel::run : Failed to run model. Message:\n\n"<<TF_Message(status)<<"\n\n")
  }
  TF_DeleteStatus(status);

  // Read from the output tensors array
  for(unsigned int i=0; i<outputs_.size(); ++i){
    MLC_ASSERT(output_tensors[i] != nullptr,
              "TensorFlowModel::run : Output tensor "<<i<<" is null. This means the model failed to run.")
    outputs_[i]->setTensor(output_tensors[i]);
  }

}

template<typename T>
void
TensorFlowModel::
setInputData(const std::string & name,
             const T * src)
{
  getInputTensor(name).copyIntoTensor(src);
}

template<typename T>
void
TensorFlowModel::
getOutputData(const std::string & name,
              T * dest) const
{
  getOutputTensor(name).copyFromTensor(dest);
}


const std::type_info &
TensorFlowModel::
getInputType(const std::string & name) const
{
  return getInputTensor(name).getType();
}

const std::type_info &
TensorFlowModel::
getOutputType(const std::string & name) const
{
  return getOutputTensor(name).getType();
}

const TensorFlowGraph &
TensorFlowModel::
getGraph() const
{
  MLC_ASSERT(graph_,
            "TensorFlowModel::getInputTensor : Graph has not been initialized.");
  return *graph_;
}

}

std::ostream &
operator<<(std::ostream & os,
           const mlc::TensorFlowModel & model)
{
  os << "Inputs: ";
  for(const auto & name : model.getInputNames())
    os << "\t" << name << ": " << toString(model.getInputExtent(name)) <<"\n";
  os << "Outputs:\n";
  for(const auto & name : model.getOutputNames())
    os << "\t" << name << ": " << toString(model.getOutputExtent(name)) <<"\n";
  return os;
}

#define ETI_SETGET(type) \
template void mlc::TensorFlowModel::getOutputData<type>(const std::string &, type *) const; \
template void mlc::TensorFlowModel::setInputData<type>(const std::string &, const type *);

ETI_SETGET(float)
ETI_SETGET(double)
ETI_SETGET(int)
ETI_SETGET(unsigned int)
