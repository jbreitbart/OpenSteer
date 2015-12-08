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
 * Timer to measure the time from its creation or the last restart to whenever @c elapsed_time is
 * called.
 */

#ifndef KAPAGA_kapaga_timer_H
#define KAPAGA_kapaga_timer_H

namespace kapaga {

	
	
	/**
	 * Measures the wall clock time between its creation or the last restart and calls to 
	 * @c elapsed_time.
	 *
	 * @c TimePeriod represents a time period. It needs to provide:
	 * - a default constructor, 
	 * - a copy constructor, 
	 * - a @c -= operation,
	 * - a @c static @c now member function that returns the current time period measuared from a 
	 *   certain point in time (that stays constant while the application is running).
	 * 
	 * To get the actual duration of the elapsed time you need a conversion function for the
	 * @c TimePeriod type.
	 */
	template< class TimePeriod >
	class timer {
	public:
		typedef TimePeriod time_type;
		
		/**
		 * Creates the timer and immediatly starts to measure the time.
		 */
		timer();
		
		/**
		 * Restarts the time measurement.
		 */
		void restart();
		
		/**
		 * @return the elapsed time till the creation or a restart of the timer.
		 */
		time_type const& elapsed_time() const;
		
	private:
		Time time_stamp_
		
	}; // class timer
	
} // namespace kapaga

#endif // KAPAGA_kapaga_timer_H
