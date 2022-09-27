
function(add_unit_test TEST_NAME TEST_FILENAME)

add_executable(${TEST_FILENAME} ${TEST_FILENAME}.cpp)
target_link_libraries(${TEST_FILENAME} ${LIBRARY_NAME})
add_test(Unit::${TEST_NAME} ${TEST_FILENAME})

endfunction(add_unit_test)