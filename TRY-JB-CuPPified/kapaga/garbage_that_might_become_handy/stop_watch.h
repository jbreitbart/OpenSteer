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
 * Two stop watches for wall clock time. One extremely simple and one which can be paused.
 *
 * @todo Implement not only for OpenMP!!!
 * @todo Write unit test.
 * @todo How to have the same interface for omp_get_wtime and gettimeofday?? Differentiate between
 *       @c time type and @c time_in_ms type.
 * @todo Extract timer header.
 * @todo Add @c elapsed_time_ms, @c elapsed_time_s, @c elapsed_time_m, @c elapsed_time_h,
 *       @c elapsed_time_d?
 */

#ifndef KAPAGA_kapaga_stop_watch_H
#define KAPAGA_kapaga_stop_watch_H


// Include kapaga::timer, kapaga::time_unit
#include "kapaga/timer.h"




namespace kapaga {
	

	/**
	 * Stop watch to measure time which can be restarter, suspended, and resumed.
	 *
	 * @todo Rewrite to use @c timer.
	 */
	class stop_watch {
	private:
		typedef timer::time_type time_type;
		typedef ??? elapsed_time_type;
		
		/**
		 * Constructs and immediately starts a @c stop_watch.
		 */
		stop_watch();
		
		/**
		 * Resets the @c stop_watch and immediately starts the next time measurement.
		 * Resumed the time measurement after resetting it if the stop watch was suspended/paused.
		 */
		void restart();
		
		/**
		 * Suspends/pauses time measurement until @c resume or @c restart is called. Otherwise 
		 * won't do anything.
		 */
		void suspend();
		
		/**
		 * Resumes time measurement if the stop watch was suspended/paused.
		 */
		void resume();
		
		/**
		 * @return @c true if time is measured, @c false if the stop watch is suspended/paused.
		 */
		bool running() const;
		
		/**
		 * How much time has elapsed since the construction or the last restart of the stop watch.
		 * If the stop watch is suspended/paused the time that has elasped till the suspension is
		 * returned.s
		 */
		time_type elapsed_time() const;
		
		// @todo Implement!
		static time_unit elapsed_time_time_unit() const;
		static time_unit internal_time_unit() const;
		
	private:
		time_type time_stamp_;
		time_type elapsed_time_;
		bool running_;
	}; // class stop_watch
	
	
	
} // namespace kapaga


#endif // KAPAGA_kapaga_stop_watch_H 
