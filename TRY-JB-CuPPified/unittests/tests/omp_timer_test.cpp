/**
 * Kapaga: Kassel Parallel Games
 *
 * Copyright (c) 2006-2007, Kapaga Development Group
 * All rights reserved.
 *
 * This file is part of the Kapaga project.
 * For conditions of distribution and use, see copyright notice in kapaga_license.txt.
 */

/**
 * @file
 *
 * Unit test for @c omp_timer.
 */

#include <UnitTest++/UnitTest++.h>


// Include kapaga::omp_timer, kapaga::convert_time_to_ms
#include "kapaga/omp_timer.h"


// Include kapaga::os_process_sleep_us_fast
#include "kapaga/os_process_utilities.h"



namespace {
	
	// @attention Mustn't be greater than @c kapaga::os_process_sleep_us_max!
	kapaga::omp_timer::time_type const sleep_ms = 1000;
	kapaga::omp_timer::time_type const time_epsilon_ms = 1;
	
} // anonymous namespace

SUITE(omp_timer_test)
{
	
	TEST(convert_time_to_ms)
	{
		CHECK_EQUAL( 1000.0, kapaga::convert_time_to_ms< double >( 1.0 ) );
	}
	

	TEST(elapsed_time)
	{
		using namespace kapaga;
		
		omp_timer timer;
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( sleep_ms ) );
		CHECK_CLOSE( sleep_ms, convert_time_to_ms< double >( timer.elapsed_time() ), time_epsilon_ms );
		
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( sleep_ms ) );
		CHECK_CLOSE( sleep_ms + sleep_ms, convert_time_to_ms< double >( timer.elapsed_time() ), time_epsilon_ms );
	}

	TEST(restart)
	{
		using namespace kapaga;
		
		omp_timer timer;
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( sleep_ms ) );
		CHECK_CLOSE( sleep_ms, convert_time_to_ms< double >( timer.elapsed_time() ), time_epsilon_ms );
		
		timer.restart();
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( sleep_ms ) );
		CHECK_CLOSE( sleep_ms, convert_time_to_ms< double >( timer.elapsed_time() ), time_epsilon_ms );
	}
	
	
} // SUITE(omp_timer_test)


