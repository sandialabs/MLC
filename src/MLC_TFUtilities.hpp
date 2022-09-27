#ifndef MLC_TFUtilities_hpp_
#define MLC_TFUtilities_hpp_

// STL includes
#include <string>
#include <vector>
#include <sstream>
#include <typeinfo>

// TensorFlow includes
#include "tensorflow/c/c_api.h"

namespace mlc {

/// Convert a TF data type to a string (for debugging)
std::string
typeToString(const TF_DataType & data_type);

/// Convert a C++ data type to a TF data type
TF_DataType
getDataType(const std::type_info & type);

/// Convert a TF data type to a C++ data type
const std::type_info &
getTypeInfo(const TF_DataType & data_type);

/// Get the tensor dimensions for a given entry in the graph
std::vector<int>
getDims(TF_Graph * graph,
        const TF_Output & field);

/// Write out the inputs for a given operation (i.e. node) in the graph
void
printInputs(std::ostream & os,
            TF_Graph * graph,
            TF_Operation * op,
            const std::string & indent = "");

/// Write out the outputs for a given operation (i.e. node) in the graph
void
printOutputs(std::ostream & os,
             TF_Graph * graph,
             TF_Operation * op,
             const std::string & indent = "");

/// Write out the entire graph
void
printGraph(std::ostream & os,
           TF_Graph * graph);

}

#endif
