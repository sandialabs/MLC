#ifndef MLC_UnitTestHarness_hpp_
#define MLC_UnitTestHarness_hpp_

#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <memory>
#include <cmath>

#include "MLC_Utilities.hpp"

class
Test
{
public:
  Test(const std::string & group, const std::string & name): name_(group+"_"+name){}
  virtual ~Test() = default;
  bool run(std::ostream & out) const {
    try{
        runImpl(out);
    } catch (const std::exception & e) {
      out << "*** Test Failed ***\n\n" << e.what() <<
      "\n\n*******************\n\n";
      return false;
    }
    return true;
  }
  const std::string & name() const {return name_;}

  virtual void runImpl(std::ostream & out) const = 0;
protected:
  std::string name_;
};

std::vector<std::unique_ptr<Test>> __tests__;

template<typename T>
struct
TestInitializer
{TestInitializer() {__tests__.push_back(std::unique_ptr<Test>(new T));}};

int
main(int arc, char** argv)
{

  std::ostream & out = std::cout;
  bool success = true;

  for(const auto & test : __tests__){
    out << "===  Running Test "<<test->name()<<"  ===\n\n";
    success = success and test->run(out);
    out << "=== Completed Test "<<test->name()<<" ===\n\n";
  }

  if(success)
    return EXIT_SUCCESS;
  return EXIT_FAILURE;

}

#define ADD_TEST(test_group, test_name) \
class Test_##test_group##_##test_name: public Test { public: \
  Test_##test_group##_##test_name(): Test(#test_group, #test_name) {} \
  void runImpl(std::ostream & out) const override; \
}; \
TestInitializer<Test_##test_group##_##test_name> initialize_Test_##test_group##_##test_name; \
void Test_##test_group##_##test_name::runImpl(std::ostream & out) const

#define ASSERT(value) \
    if(not (value)){ \
      std::stringstream __ss; \
      __ss << "ERROR in "<<__FILE__<<":"<<__LINE__<<" :\n\nAssertion Failed : '" << #value << "' evaluated to " << bool(value) << ".\n\n"; \
      throw std::logic_error(__ss.str()); \
    } else { \
      out << "Assertion Passed : '" << #value << "' evaluated to "<<bool(value)<<".\n"; \
    }

#define ASSERT_LT(a,b) \
    if(not ((a) < (b))){ \
      std::stringstream __ss; \
      __ss << "ERROR in "<<__FILE__<<":"<<__LINE__<<" :\n\nAssertion Failed : '" << #a << "' (" << a << ") is not less than '" << #b << "' (" << b << ").\n\n"; \
      throw std::logic_error(__ss.str()); \
    } else { \
      out << "Assertion Passed : '" << #a << "' (" << a << ") is less than '" << #b << "' (" << b << ").\n"; \
    }

#define ASSERT_EQ(a,b) \
    if(a != b){ \
      std::stringstream __ss; \
      __ss << "ERROR in "<<__FILE__<<":"<<__LINE__<<" :\n\nEquality Assertion Failed : " << #a << " == " << #b << " evaluated to " << a << " == " << b << " which is false.\n\n"; \
      throw std::logic_error(__ss.str()); \
    } else { \
      out << "Equality Assertion Passed : " << #a << " == " << #b << " evaluated to " << a << " == " << b << " which is true.\n"; \
    }

#define ASSERT_NEARLY_EQ(a,b,tol) \
    if(isnan(a) or isnan(b) or isnan(tol)){ \
      std::stringstream __ss; \
      __ss << "ERROR in "<<__FILE__<<":"<<__LINE__<<" :\n\nNear Equality Assertion Failed (NaN Found): " << #a << " ~= " << #b << " evaluated to (" << a << ") == (" << b << ") (tolerance "<<tol<<").\n\n"; \
      throw std::logic_error(__ss.str()); \
    } else if(std::fabs((a)-(b)) > tol){ \
      std::stringstream __ss; \
      __ss << "ERROR in "<<__FILE__<<":"<<__LINE__<<" :\n\nNear Equality Assertion Failed : " << #a << " ~= " << #b << " evaluated to (" << a << ") == (" << b << ") (tolerance "<<tol<<") which is false.\n\n"; \
      throw std::logic_error(__ss.str()); \
    } else { \
      out << "Near Equality Assertion Passed : " << #a << " ~= " << #b << " evaluated to (" << a << ") == (" << b << ") (tolerance "<<tol<<") which is true.\n"; \
    }

#define ASSERT_THROWS(cmd) \
    {bool threw = false; \
    try{ \
      cmd; \
    } catch (const std::exception & __e) { \
      out << "Exception test passed. Exception:\n\n" << __e.what() << "\n\n"; \
      threw = true; \
    } catch (...) { \
      out << "Exception test passed with unknown Exception.\n"; \
      threw = true; \
    } \
    if(not threw) { \
      std::stringstream __ss; \
      __ss << "ERROR in "<<__FILE__<<":"<<__LINE__<<" :\n\nCommand did not throw.\n\n"; \
      throw std::logic_error(__ss.str()); \
    }}

#define ASSERT_IN(a,vec) \
    { \
      bool __found = false; \
      for(const auto & __v : vec){ \
        if(__v == a){ __found = true; break; } \
      } \
      if(not __found){ \
        std::stringstream __ss; \
        __ss << "ERROR in "<<__FILE__<<":"<<__LINE__<<" :\n\nIn Assertion Failed : Quantity "<<#a<<" (evaluates to "<<a<<" not found in "<<#vec<<".\n\n"; \
        throw std::logic_error(__ss.str()); \
      } else { \
        out << "In Assertion Passed : " << #a << " in " << #vec << ".\n"; \
      } \
    }


#endif

