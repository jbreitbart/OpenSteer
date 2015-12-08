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
 * Unit test for @c omp_stop_watch.
 */

#include <UnitTest++/UnitTest++.h>

// Include kapaga::omp_stop_watch
#include "kapaga/omp_stop_watch.h"

// Include kapaga::omp_timer
#include "kapaga/omp_timer.h"

// Include kapaga::os_process_sleep_ms
#include "kapaga/os_process_utilities.h"

// Include kapaga::duration, kapaga::omp_time_t, kapaga::omp_time_conversion_s_to_ms


namespace {
	
	double const time_comparison_epsilon_ms = 1.0;
	double const time_comparison_epsilon_s  = 0.1;
	
} // anonymous namespace


SUITE(omp_stop_watch_test)
{
	
	
	
	
	TEST(elapsed_time)
	{
		// Test that a stop watch and a timer are near each other when measuring elasped time
		// without any pauses or restarts.
		// As @c kapaga::os_process_sleep_ms is used this also tests if the stop watch measures
		// wall clock time or cpu ticks (it should/must measure wall clock time).
		
		using namespace kapaga;
		
		omp_stop_watch watch;
		omp_timer timer;
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );
		omp_stop_watch::time_type stop_watch_time = watch.elapsed_time();
		omp_timer::time_type timer_time = timer.elapsed_time();
		
		double stop_watch_ms = convert_time_to_ms< double >( stop_watch_time );
		double timer_ms = convert_time_to_ms< double >( timer_time );
		
		CHECK_CLOSE( timer_ms, stop_watch_ms, time_comparison_epsilon_ms );
		
	}
	
	
	TEST(suspend_and_resume)
	{
		// Check that a stop watch doesn't measure time while suspended
		
		using namespace kapaga;
		
		
		omp_stop_watch watch;
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );
		omp_stop_watch::time_type time0 = watch.elapsed_time();
		watch.suspend();
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );
		omp_stop_watch::time_type time1 = watch.elapsed_time();
		
		CHECK_CLOSE( convert_time_to_ms< double >( time0 ), 
					 convert_time_to_ms< double >( time1 ), 
					 static_cast< double >( time_comparison_epsilon_ms ) );
		
		watch.resume();
		omp_timer timer2;
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );
		omp_stop_watch::time_type watch_time2 = watch.elapsed_time();
		omp_timer::time_type timer2_time = timer2.elapsed_time();
		
		CHECK_CLOSE( convert_time_to_ms< double >( time1 ) + convert_time_to_ms< double >( timer2_time ),
					 convert_time_to_ms< double >( watch_time2 ), time_comparison_epsilon_ms );
	}
	
	TEST(restart_while_running)
	{
		using namespace kapaga;
		
		// Restart while running.
		omp_stop_watch watch;
		omp_timer timer;
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );
		watch.restart();
		timer.restart();
		omp_stop_watch::time_type watch_time0 = watch.elapsed_time();
		omp_timer::time_type timer_time0 = timer.elapsed_time();
		
		CHECK_CLOSE( convert_time_to_ms< double >( timer_time0 ),
					 convert_time_to_ms< double >( watch_time0 ),
					 time_comparison_epsilon_ms );
				
	}	
		
	TEST(restart_while_suspended)
	{
		using namespace kapaga;
		
		// Restart while running.
		omp_stop_watch watch;
		omp_timer timer;
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );
		watch.suspend();
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );		
		watch.restart();
		timer.restart();
		omp_stop_watch::time_type watch_time0 = watch.elapsed_time();
		omp_timer::time_type timer_time0 = timer.elapsed_time();
		
		CHECK_CLOSE( convert_time_to_ms< double >( timer_time0 ),
					 convert_time_to_ms< double >( watch_time0 ),
					 time_comparison_epsilon_ms );
		
	}	
	
	
	
	TEST(start_suspended)
	{
		using namespace kapaga;
		
		omp_stop_watch watch( omp_stop_watch::start_suspended );
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );
		omp_timer timer;
		watch.resume();
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );
		omp_timer::time_type timer_time = timer.elapsed_time();
		omp_stop_watch::time_type watch_time = watch.elapsed_time();
		
		CHECK_CLOSE( convert_time_to_ms< double >( timer_time ),
					 convert_time_to_ms< double >( watch_time ),
					 time_comparison_epsilon_ms );
		
	}
	
	
	TEST(restart_suspended)
	{
		using namespace kapaga;
		
		omp_stop_watch watch;
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );
		watch.restart( omp_stop_watch::start_suspended );
		omp_stop_watch other_watch( omp_stop_watch::start_suspended );
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );
		CHECK_CLOSE( convert_time_to_ms< double >( watch.elapsed_time() ),
					 convert_time_to_ms< double >( other_watch.elapsed_time() ),
					 time_comparison_epsilon_ms );
	}
	
	TEST(restart_suspended_while_suspended)
	{
		using namespace kapaga;
		
		omp_stop_watch watch;
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );
		watch.suspend();
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );
		watch.restart( omp_stop_watch::start_suspended );
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );
		watch.resume();
		omp_timer timer;
		os_process_sleep_ms( static_cast< os_process_sleep_ms_t >( 1000 ) );	
		omp_stop_watch::time_type watch_time = watch.elapsed_time();
		omp_timer::time_type timer_time = timer.elapsed_time();
		
		CHECK_CLOSE( convert_time_to_ms< double >( timer_time ),
					 convert_time_to_ms< double >( watch_time ),
					 time_comparison_epsilon_ms );
		
	}
	
} // SUITE(omp_stop_watch_test)
