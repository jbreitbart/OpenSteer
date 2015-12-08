/**
 * Kapaga: Kassel Parallel Games
 *
 * Copyright (c) 2006-2007, Kapaga Development Group
 * All rights reserved.
 *
 * This file is part of the Kapaga project.
 * For conditions of distribution and use, see copyright notice in kapaga_license.txt.
 */

#include "kapaga/omp_time_utilities.h"


kapaga::omp_time_t const kapaga::omp_time_factor_s_to_ms = static_cast< kapaga::omp_time_t >( 1000 );

kapaga::omp_time_t const kapaga::omp_time_factor_ms_to_us = static_cast< kapaga::omp_time_t >( 1000 );

kapaga::omp_time_t const kapaga::omp_time_factor_s_to_us = kapaga::omp_time_factor_s_to_ms * kapaga::omp_time_factor_ms_to_us;


kapaga::omp_time_t const kapaga::omp_time_factor_us_to_ms = static_cast< kapaga::omp_time_t >( 1 ) / kapaga::omp_time_factor_ms_to_us;

kapaga::omp_time_t const kapaga::omp_time_factor_ms_to_s = static_cast< kapaga::omp_time_t >( 1 ) / kapaga::omp_time_factor_s_to_ms;

kapaga::omp_time_t const kapaga::omp_time_factor_us_to_s = static_cast< kapaga::omp_time_t >( 1 ) / kapaga::omp_time_factor_s_to_us;

