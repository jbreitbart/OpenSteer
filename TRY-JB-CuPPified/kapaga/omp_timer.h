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
 * @file omp_timer.h
 *
 * Timer that starts to measure the elapsed wall clock time when created and can be restarted.
 */

//Header guard macro format: KAPAGA_namespace_file_name_H
#ifndef KAPAGA_kapaga_omp_timer_H
#define KAPAGA_kapaga_omp_timer_H

// Include omp_get_wtime
#include "kapaga/omp_header_wrapper.h"


// Include kapaga::omp_time_t
#include "kapaga/omp_time_utilities.h"


namespace kapaga {
	
	/**
	 * Timer using @c omp_get_wtime to measure the elapsed wall clock time since its construction
	 * or the last restart.
	 *
	 * Time measurement can't be stopped or paused. It is intended to measure only short time
	 * periods and it isn't a high precision timer.
	 * 
	 * Use the @c convert_time_to helper functions from @c omp_time_utilities.h to get 
	 * access to the elasped time in a specified time unit.
	 *
	 * @attention Isn't thread safe. Use an instance only in the thread that created it.
	 * @attention Isn't consistent across different threads. Only measure the elapsed time and
	 *            restart it from the thread that constructed it.
	 */
	class omp_timer {
	public:
		// @attention Only use the typedef'ed type!
		typedef omp_time_t time_type;
		
		/**
		 * Constructs the timer and starts time measurement.
		 */
		omp_timer() : time_stamp_( omp_get_wtime() ) 
		{
			// Nothing to do.
		}
		
		/**
		 * Restart the timer.
		 */
		void restart()
		{
			time_stamp_ = omp_get_wtime();
		}
		
		/**
		 * @return the elapsed time since construction or the last restart.
		 *
		 * @attention @c omp_timer doesn't guarantee what time unit is used. Call
		 *            @c convert_time_to_ms to get the time in milliseconds or
		 *            @c convert_time_to_s to the the time in seconds.
		 */
		time_type elapsed_time() const
		{
			return omp_get_wtime() - time_stamp_;
		}
		
	private:
		time_type time_stamp_;
	}; // class omp_timer
	
	
	

	
	
} // namespace kapaga


#endif // KAPAGA_kapaga_omp_timer_H
