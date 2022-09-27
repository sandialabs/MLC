#include "MLC_TFUtilities.hpp"

// Internal includes
#include "MLC_Utilities.hpp"
#include "MLC_Exceptions.hpp"

namespace mlc {

std::string
typeToString(const TF_DataType & data_type) {
  switch (data_type) {
    case TF_FLOAT:
      return "TF_FLOAT";
    case TF_DOUBLE:
      return "TF_DOUBLE";
    case TF_INT32:
      return "TF_INT32";
    case TF_UINT8:
      return "TF_UINT8";
    case TF_INT16:
      return "TF_INT16";
    case TF_INT8:
      return "TF_INT8";
    case TF_STRING:
      return "TF_STRING";
    case TF_COMPLEX64:
      return "TF_COMPLEX64";
    case TF_INT64:
      return "TF_INT64";
    case TF_BOOL:
      return "TF_BOOL";
    case TF_QINT8:
      return "TF_QINT8";
    case TF_QUINT8:
      return "TF_QUINT8";
    case TF_QINT32:
      return "TF_QINT32";
    case TF_BFLOAT16:
      return "TF_BFLOAT16";
    case TF_QINT16:
      return "TF_QINT16";
    case TF_QUINT16:
      return "TF_QUINT16";
    case TF_UINT16:
      return "TF_UINT16";
    case TF_COMPLEX128:
      return "TF_COMPLEX128";
    case TF_HALF:
      return "TF_HALF";
    case TF_RESOURCE:
      return "TF_RESOURCE";
    case TF_VARIANT:
      return "TF_VARIANT";
    case TF_UINT32:
      return "TF_UINT32";
    case TF_UINT64:
      return "TF_UINT64";
    default:
      return "Unknown";
  }
}


TF_DataType
getDataType(const std::type_info & type)
{
  if(type == typeid(float))
    return TF_FLOAT;
  else if(type == typeid(double))
    return TF_DOUBLE;
  else if(type == typeid(int))
    return TF_INT32;
  else if(type == typeid(long int))
    return TF_INT64;
  else if(type == typeid(unsigned int))
    return TF_UINT32;
  else if(type == typeid(unsigned long int))
    return TF_UINT64;
  else {
    MLC_THROW("getDataType : Unknown tensorflow type for " << demangleTypeInfoName(type));
  }

}

const std::type_info &
getTypeInfo(const TF_DataType & data_type)
{
  switch (data_type) {
    case TF_FLOAT:
      return typeid(float);
    case TF_DOUBLE:
      return typeid(double);
    case TF_INT32:
      return typeid(int);
    case TF_INT64:
      return typeid(long int);
    case TF_UINT32:
      return typeid(unsigned int);
    case TF_UINT64:
      return typeid(unsigned long int);
    default:
      MLC_THROW("getType: No support for " << typeToString(data_type));
  }
}


std::vector<int>
getDims(TF_Graph * graph,
        const TF_Output & field)
{
  TF_Status * status = TF_NewStatus();

  const int num_dims = TF_GraphGetTensorNumDims(graph, field, status);

  if(num_dims <= 0)
    return {};

  if (TF_GetCode(status) != TF_OK) {
    TF_DeleteStatus(status);
    MLC_THROW("getDims: Can't get tensor dimensionality")
  }

  std::vector<std::int64_t> dims(num_dims);
  TF_GraphGetTensorShape(graph, field, dims.data(), num_dims, status);

  if (TF_GetCode(status) != TF_OK) {
    TF_DeleteStatus(status);
    MLC_THROW("getDims: Can't get get tensor shape")
  }

  TF_DeleteStatus(status);

  std::vector<int> idims;
  for(const auto & d : dims)
    idims.push_back(d);
  return idims;
}

void
printInputs(std::ostream & os,
            TF_Graph * graph,
            TF_Operation * op,
            const std::string & indent)
{
  const int num_inputs = TF_OperationNumInputs(op);

  for (int i=0; i<num_inputs; ++i) {
    TF_Input input{op, i};
    TF_DataType type = TF_OperationInputType(input);
    os << indent << std::to_string(i) << ": " << typeToString(type);

    TF_Output output = TF_OperationInput(input);
    const auto dims = getDims(graph,output);

    const std::string output_op_name = TF_OperationName(output.oper);

    os << " - from " << output_op_name;
    if(dims.size() > 0)
      os << ", dims "<<toString(dims);

    os <<"\n";

  }
}

void
printOutputs(std::ostream & os,
             TF_Graph * graph,
             TF_Operation * op,
             const std::string & indent)
{
  const int num_outputs = TF_OperationNumOutputs(op);

  TF_Status * status = TF_NewStatus();

  for(int i=0; i<num_outputs; ++i) {

    TF_Output output{op, i};
    const TF_DataType type = TF_OperationOutputType(output);

    os << indent << std::to_string(i) << ": " << typeToString(type);

    const auto dims = getDims(graph, output);
    const int num_dims = dims.size();

    if(dims.size() > 0)
      os << ", dims " << toString(dims);
    os << "\n";
  }

  TF_DeleteStatus(status);
}


void
printGraph(std::ostream & os,
           TF_Graph * graph)
{
  TF_Status * status = TF_NewStatus();

  std::size_t pos = 0;
  TF_Operation * op = nullptr;

  // Iterate through operations in the graph
  while( (op = TF_GraphNextOperation(graph, &pos)) != nullptr){

    const std::string name = TF_OperationName(op);
    const std::string type = TF_OperationOpType(op);
    const std::string device = TF_OperationDevice(op);

    const int num_outputs = TF_OperationNumOutputs(op);
    const int num_inputs = TF_OperationNumInputs(op);

    os << "Operation " << pos << " (" << name << "): type: " << type << ", device: " << device << ", inputs: " << num_inputs << ", outputs: " << num_outputs << "\n";
    if(num_inputs > 0){
      os << "  Inputs:\n";
      printInputs(os,graph,op,"    ");
    }
    if(num_outputs > 0){
      os << "  Outputs:\n";
      printOutputs(os,graph,op,"    ");
    }

  }

  TF_DeleteStatus(status);
}

}
