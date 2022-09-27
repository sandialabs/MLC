#ifndef MLC_Exceptions_hpp_
#define MLC_Exceptions_hpp_

// STL includes
#include <sstream>
#include <stdexcept>

#define MLC_ASSERT(assertion, msg) \
  if(not (assertion)){ \
    std::stringstream assertion_ss; \
    assertion_ss << msg; \
    throw std::logic_error(assertion_ss.str()); \
  }

#define MLC_THROW(msg) \
  { \
    std::stringstream throw_ss; \
    throw_ss << msg; \
    throw std::logic_error(throw_ss.str()); \
  }


#endif
