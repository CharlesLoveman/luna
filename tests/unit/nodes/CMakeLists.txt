# tests/unit/nodes/CMakeLists.txt
# Node unit tests

set(NODES_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(add_node_test TEST_NAME)
    luna_add_test(${TEST_NAME} ${TEST_NAME}.cpp "nodes" "unit")
endfunction()

# Node tests
add_node_test(test_relu_node)
add_node_test(test_linear_node)
add_node_test(test_conv_node)
add_node_test(test_batchnorm_node)
add_node_test(test_add_node)
add_node_test(test_sigmoid_node)
add_node_test(test_softmax_node)
add_node_test(test_flatten_node)
add_node_test(test_reshape_node)
