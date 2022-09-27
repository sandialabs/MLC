include(AddUnitTest)

#####################
# Moments tests

add_unit_test(Scaling scaling)
add_unit_test(MomentIndexer moment_indexer)
add_unit_test(MomentConversion moment_conversion)
add_unit_test(TensorFlowModel tensorflow_model)
add_unit_test(CollisionModel collision_model)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/example_model 
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
)

#####################
