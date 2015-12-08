/*
 *  timer.cpp
 *  OpenSteer
 *
 *  Created by Björn Knafla on 30.01.07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#include "timer.h"



kapaga::timer::timer()
: time_stamp_( TimePeriod::now() )
{
	// Nothing to do.
}



void 
kapaga::timer::restart()
{
	time_stamp_ = time_type::now();
}



kapaga::timer::time_type const&
kapaga::timer::elapsed_time() const
{
	return time_type() -= time_stamp_;
}
