#ifndef MLC_Utilities_hpp_
#define MLC_Utilities_hpp_

// STL includes
#include <string>
#include <vector>
#include <sstream>
#include <typeinfo>

/// Get a string for a given type_info object (hopefully more advanced than info.name())
std::string
demangleTypeInfoName(const std::type_info & info);

/// Convert a vector to a string
template<typename T>
std::string
toString(const std::vector<T> & vector)
{
  std::stringstream ss;
  ss << "[";
  for(unsigned int i=0; i<vector.size(); ++i){
    ss << vector[i];
    if(i != vector.size()-1)
      ss << ", ";
  }
  ss << "]";
  return ss.str();
}

#endif
