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
 * Unit test for @c omp_stubs.
 *
 * @attention Only call these tests when <em>not</em> compiling with OpenMP!
 *
 * @todo Put all time related tests into their own project so they don't slow down unit testing for
 *       the other project though calling sleep or running in loops for nothing.
 * @todo Rewrite tests to not use @c std::clock - this breaks all the time when compared to real
 *       wall clock time.
 */

#include <UnitTest++/UnitTest++.h>

// Include omp_get_wtick, omp_get_wtime, omp_get_max_threads, omp_get_num_threads, omp_get_thread_num
#include "omp_stubs.h"


// Include std::clock, CLOCKS_PER_SEC
#include <ctime>


// Include kapaga::os_process_sleep_ms, kapaga::os_process_sleep_ms_t
#include "kapaga/os_process_utilities.h"


namespace {
	
	// Sleep for this amount of milliseconds to see if @c omp_get_wtime measures clock ticks that
	// aren't counted during sleep, or real wall clock time.
	kapaga::os_process_sleep_ms_t const sleep_for_ms = 10000;
	double const sleep_ms_in_s = static_cast< double >( sleep_for_ms ) / 1000.0;
	
	// Loop count to test @c omp_get_wtime.
	int const test_wtime_loop_count = 100000000;
	
	// Epsilon to measure that @c std::clock and @c omp_get_wtime are close each other.
	double const epsilon = 1.0 / CLOCKS_PER_SEC;
	
} // anonymous namespace



SUITE(omp_stubs_test)
{
	
	TEST(omp_get_wtime_measures_wall_clock_time)
	{
		// Check if @c omp_get_wtime from the stubs really measures wall clock time and not just 
		// CPU ticks that don't get counted if the process or the thread sleeps.
		
		
		double const start_time = omp_get_wtime();
		kapaga::os_process_sleep_ms( sleep_for_ms );
		double const end_time = omp_get_wtime();
		
		CHECK_CLOSE( sleep_ms_in_s, end_time - start_time, epsilon );
	
	}

	
	TEST(omp_get_wtick) 
	{
		CHECK_EQUAL( 1.0 / static_cast< double >( CLOCKS_PER_SEC), omp_get_wtick() );
	}
	
	TEST(omp_get_wtime) 
	{
		// Idea: measure the time a loop takes with std::clock and omp_get_wtime and compare if the
		//       results are close to each other.
		//       This isn't a really high quality test.
		// @todo Rewrite test to use wall clock time for comparison with @c omp_get_wtime!
		//
		// @ATTENTION THis test might fail because of thedifference between wall clock time and
		//            cpu tick time!
		
		std::clock_t start_clock = std::clock();
		double start_wtime = omp_get_wtime();
		
		int count_temp = 0;
		int const max_count = test_wtime_loop_count;
		for ( int i = 0; i < max_count; ++i ) {
			count_temp += i;
		}
		
		std::clock_t end_clock = std::clock();
		double end_wtime = omp_get_wtime();
		
		double const eps = epsilon;
		double time_dif_clock = static_cast< double >( end_clock - start_clock ) / static_cast< double >( CLOCKS_PER_SEC );
		double time_dif_wtime = end_wtime - start_wtime;
		
		CHECK_CLOSE( time_dif_clock, time_dif_wtime, eps );
		
	}

	TEST(omp_get_max_threads)
	{
		CHECK_EQUAL( 1, omp_get_max_threads() );
	}

	TEST(omp_get_num_threads)
	{
		CHECK_EQUAL( 1, omp_get_num_threads() );
	}

	TEST(omp_get_thread_num)
	{
		CHECK_EQUAL( 0, omp_get_thread_num() );
	}
	
} // SUITE(omp_stubs_test)

