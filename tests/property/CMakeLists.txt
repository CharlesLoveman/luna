# tests/property/CMakeLists.txt
# Property-based test configuration

set(PROPERTY_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(add_property_test TEST_NAME)
    luna_add_test(${TEST_NAME} ${TEST_NAME}.cpp "property" "unit")
endfunction()

add_property_test(test_bound_monotonicity)
add_property_test(test_alpha_convergence)
add_property_test(test_random_networks)
add_property_test(test_batch_invariance)
add_property_test(test_batch_invariance_conv)
add_property_test(test_batch_invariance_ibp)
add_property_test(test_batch_search_state)
