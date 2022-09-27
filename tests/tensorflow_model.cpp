
#include "UnitTestHarness.hpp"

#include "MLC_TensorFlowGraph.hpp"
#include "MLC_TensorFlowModel.hpp"

ADD_TEST(TensorFlow,Loader)
{

  using namespace mlc;

   TensorFlowModel model("example_model/model");

   std::cout << "===========================================================================================\n";
   model.getGraph().toStream(std::cout);
   std::cout << "===========================================================================================\n";
   std::cout << model.getGraph();
   std::cout << "===========================================================================================\n";
   std::cout << "Layer 0:\n";
   std::cout << "Weights:\n";
   std::cout << model.getGraph().getLayerWeights<float>("HiddenBlock0");
   std::cout << "Bias:\n";
   std::cout << model.getGraph().getLayerBias<float>("HiddenBlock0") << "\n";
   std::cout << "===========================================================================================\n";
   std::cout << "Layer 1:\n";
   std::cout << "Weights:\n";
   std::cout << model.getGraph().getLayerWeights<float>("HiddenBlock1");
   std::cout << "Bias:\n";
   std::cout << model.getGraph().getLayerBias<float>("HiddenBlock1") << "\n";
   std::cout << "===========================================================================================\n";
   std::cout << "dG:\n";
   std::cout << "Weights:\n";
   std::cout << model.getGraph().getLayerWeights<float>("dG");
   std::cout << "Bias:\n";
   std::cout << model.getGraph().getLayerBias<float>("dG") << "\n";
   std::cout << "===========================================================================================\n";
   std::cout << "tau:\n";
   std::cout << "Weights:\n";
   std::cout << model.getGraph().getLayerWeights<float>("tau");
   std::cout << "Bias:\n";
   std::cout << model.getGraph().getLayerBias<float>("tau") << "\n";
   std::cout << "===========================================================================================\n";

   const auto model_inputs = model.getInputNames();
   const auto model_outputs = model.getOutputNames();

   const unsigned int extent_1 = 15;

   out << "Inputs:\n";
   for(const auto & name : model_inputs)
     out << "\t"<<name<<"\n";

   out << "Outputs:\n";
   for(const auto & name : model_outputs)
     out << "\t"<<name<<"\n";

   const std::string input_0 = "F";
   const std::string input_1 = "n";
   const std::string input_2 = "T";

   // TODO: We really have to fix the naming here...
   const std::string output_0 = "StatefulPartitionedCall:0";
   const std::string output_1 = "StatefulPartitionedCall:1";

   ASSERT_IN(input_0, model_inputs)
   ASSERT_IN(input_1, model_inputs)
   ASSERT_IN(input_2, model_inputs)

   ASSERT_IN(output_0, model_outputs)
   ASSERT_IN(output_1, model_outputs)

   ASSERT_EQ(model.getInputExtent(input_0).size(), 2)
   ASSERT_EQ(model.getInputExtent(input_1).size(), 2)
   ASSERT_EQ(model.getInputExtent(input_2).size(), 2)

   ASSERT_EQ(model.getInputExtent(input_0)[0], 1)
   ASSERT_EQ(model.getInputExtent(input_0)[1], extent_1)

   ASSERT_EQ(model.getInputExtent(input_1)[0], 1)
   ASSERT_EQ(model.getInputExtent(input_1)[1], 1)

   ASSERT_EQ(model.getInputExtent(input_2)[0], 1)
   ASSERT_EQ(model.getInputExtent(input_2)[1], 1)

   ASSERT_EQ(model.getOutputExtent(output_0).size(), 2)
   ASSERT_EQ(model.getOutputExtent(output_1).size(), 2)

   ASSERT_EQ(model.getOutputExtent(output_0)[0], 1)
   ASSERT_EQ(model.getOutputExtent(output_0)[1], extent_1)

   ASSERT_EQ(model.getOutputExtent(output_1)[0], 1)
   ASSERT_EQ(model.getOutputExtent(output_1)[1], 1)

   ASSERT(model.getInputType(input_0) == typeid(float))
   ASSERT(model.getInputType(input_1) == typeid(float))
   ASSERT(model.getInputType(input_2) == typeid(float))

   ASSERT(model.getOutputType(output_0) == typeid(float))
   ASSERT(model.getOutputType(output_1) == typeid(float))

   // NOTE: Every time the build_test_tensorflow_model.py is run, these arrays need to be updated
   std::vector<float> input_data_0, input_data_1, input_data_2;
   for(int i=0; i<extent_1; ++i)
     input_data_0.push_back(0.2);
   input_data_1.push_back(1.e21);
   input_data_2.push_back(10.);

   std::vector<float> output_data_0, output_data_1;
   output_data_0.resize(extent_1);
   output_data_1.resize(1);

   const float tolerance = 1.e-4;

   model.setInputData(input_0,input_data_0.data());
   model.setInputData(input_1,input_data_1.data());
   model.setInputData(input_2,input_data_2.data());

   ASSERT_THROWS(model.setInputData(output_0,input_data_0.data());)

   model.run();
   model.getOutputData(output_0,output_data_0.data());
   model.getOutputData(output_1,output_data_1.data());

   // Output is meaningless - we just ran it and used those values
   ASSERT_NEARLY_EQ(output_data_0[0], 0.161153, tolerance)
   ASSERT_NEARLY_EQ(output_data_0[12], 1.62895, tolerance)
   ASSERT_NEARLY_EQ(output_data_1[0], -1.41379, tolerance);

}
