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
 * @file omp_stop_watch.h
 *
 * A stop watch.
 */

//Header guard macro format: KAPAGA_namespace_file_name_H
#ifndef KAPAGA_kapaga_omp_stop_watch_H
#define KAPAGA_kapaga_omp_stop_watch_H

// Include kapaga::omp_time_t
#include "kapaga/omp_time_utilities.h"


namespace kapaga {
	
	/**
	 * Stop watch to measure time which can be restarted, suspended, and resumed.
	 *
	 * It uses @c omp_get_wtime internally.
	 * Don't use it if high time precision, reliability, or measuring long durations is needed.
	 *
	 * Call the @c convert_time_to functions from @c omp_time_utilities.h to convert the 
	 * returned @c time_type into a specific  time unit, like seconds, milliseconds, etc.
	 *
	 * Pass @c omp_stop_watch::start_suspended to the constructor to start suspended or
	 * @c omp_stop_watch::start_running (the default constructor) to start running
	 * (immediately start measuring time).
	 *
	 * @attention Isn't thread safe. Use an instance only in the thread that created it.
	 * @attention Isn't consistent across different threads. Only measure the elapsed time and
	 *            operate it from the thread that constructed it.
	 */
	class omp_stop_watch {
	public:
		
		struct start_suspended_type {};
		static start_suspended_type const start_suspended;
		
		struct start_running_type {};
		static start_running_type const start_running;
		
		
		// @attention Only refer to @c omp_stop_watch::time_type and not to the type it is based on.
		typedef omp_time_t time_type;
		
		
		/**
		 * Constructs and immediately starts an @c omp_stop_watch.
		 */
		explicit omp_stop_watch( start_running_type const& _start_running = start_running );
		
		/**
		 * Constructs an @c omp_stop_watch in suspended mode.
		 */
		explicit omp_stop_watch( start_suspended_type const& _start_suspended );
		
		/**
		 * Resets the @c omp_stop_watch and immediately starts the next time measurement.
		 * Resumes the time measurement after resetting it if the stop watch was suspended/paused.
		 */
		void restart( start_running_type const& _start_running = start_running  );
		
		/**
		 * Resets the stop watch in suspended mode.
		 */
		void restart( start_suspended_type const& _start_suspended );
		
		/**
		 * Suspends/pauses running time measurement until @c resume or @c restart is called.
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
		 * If the stop watch is suspended/paused the time that elaspes during this mode
		 * isn't measured.
		 *
		 * Use one of the conversion functions from @c omp_time_utilities.h to convert the returned
		 * time into the needed time unit.
		 */
		time_type elapsed_time() const;
		
		
	private:
		time_type time_stamp_;
		time_type elapsed_time_;
		bool running_;
		
	}; // class omp_stop_watch
	
	
} // namespace kapaga


#endif // KAPAGA_kapaga_omp_stop_watch_H
