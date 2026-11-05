/*********************                                                        */
/*! \file LunaConfiguration.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __LunaConfiguration_h__
#define __LunaConfiguration_h__

#include "MString.h"
#include "ConfigurationError.h"
#include <torch/torch.h>

class LunaConfiguration
{
public:
    enum class AnalysisMethod
    {
        CROWN,
        AlphaCROWN
    };

    enum class BoundSide
    {
        Lower,
        Upper
    };

    static void print();
    static void resetToDefaults();
    static void parseArgs( int argc, char **argv );

    static AnalysisMethod ANALYSIS_METHOD;
    static bool COMPUTE_LOWER;
    static bool COMPUTE_UPPER;
    static bool VERBOSE;

    static unsigned ALPHA_ITERATIONS;
    /// Iterations the last alpha optimization of each side actually ran, for diagnostics.
    static unsigned LAST_LOWER_ALPHA_ITERATIONS;
    static unsigned LAST_UPPER_ALPHA_ITERATIONS;
    static float ALPHA_LR;
    /// Learning rate and step count of a call seeded with persisted alphas, such as a search node
    /// starting from its parent. Zero means the same as ALPHA_LR and ALPHA_ITERATIONS.
    static float WARM_ALPHA_LR;
    static unsigned WARM_ALPHA_ITERATIONS;
    static float ALPHA_LR_DECAY;
    static bool KEEP_BEST;
    static bool USE_SHARED_ALPHA;
    static unsigned EARLY_STOP_PATIENCE;
    static bool FIX_INTERM_BOUNDS;
    static bool STABILIZE_INTERMEDIATE_BOUNDS;
    static bool RECOMPUTE_INTERMEDIATE_BOUNDS;
    static String OPTIMIZER;
    static float START_SAVE_BEST;
    static BoundSide BOUND_SIDE;
    static bool OPTIMIZE_LOWER;
    static bool OPTIMIZE_UPPER;
    static bool STOP_CROWN_ON_VERIFIED;
    static bool STOP_ALPHA_ON_VERIFIED;

    static bool ENABLE_FIRST_LINEAR_IBP;
    static bool USE_STANDARD_CROWN;
    static bool USE_PATCHES_MODE;

    static int VERBOSITY;
    static int TIMEOUT;
    static int SEED;
    static int NUM_BLAS_THREADS;
    static String INPUT_FILE_PATH;
    static String PROPERTY_FILE_PATH;

    static bool USE_CUDA;
    static int CUDA_DEVICE_ID;
    static torch::Device DEVICE;

    static torch::Device getDevice();
    static void updateDeviceFromFlags();

    static const double DEFAULT_EPSILON_FOR_COMPARISONS;
    static const unsigned DEFAULT_DOUBLE_TO_STRING_PRECISION;
    static bool NETWORK_LEVEL_REASONER_LOGGING;
    static const double SIGMOID_CUTOFF_CONSTANT;

    static String analysisMethodToString( AnalysisMethod method );
    static AnalysisMethod stringToAnalysisMethod( const String &str );
    static String boundSideToString( BoundSide side );
    static BoundSide stringToBoundSide( const String &str );
};

#endif // __LunaConfiguration_h__
