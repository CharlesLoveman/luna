# tests/benchmarks/CMakeLists.txt
# Benchmark configuration

if(BUILD_BENCHMARKS)
    set(BENCHMARKS_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

    function(add_benchmark BENCH_NAME)
        add_executable(${BENCH_NAME} ${BENCHMARKS_TEST_DIR}/${BENCH_NAME}.cpp)

        target_link_libraries(${BENCH_NAME}
            TestFixtures
            ${LUNA_ONNX_PARSER_LIB}
            ${LUNA_LIB}
            benchmark::benchmark
            ${LIBS}
        )

        target_include_directories(${BENCH_NAME} PRIVATE
            ${PROJECT_SOURCE_DIR}  # Project root for "src/..." includes
            ${PROJECT_SOURCE_DIR}/tests  # For "fixtures/..." includes
            ${TEST_INCLUDE_DIRS}
        )

        target_compile_options(${BENCH_NAME} PRIVATE ${TEST_COMPILE_FLAGS})
    endfunction()

    add_benchmark(bench_crown_analysis)
    add_benchmark(bench_alpha_crown)
    add_benchmark(bench_ibp_vs_crown)
    add_benchmark(bench_batched_ibp)
    add_benchmark(bench_acasxu)
endif()
