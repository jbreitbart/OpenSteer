/**
 * Kapaga: Kassel Parallel Games
 *
 * Copyright (c) 2006-2007, Kapaga Development Group
 * All rights reserved.
 *
 * This file is part of the Kapaga project.
 * For conditions of distribution and use, see copyright notice in kapaga_license.txt.
 */

#include "kapaga/posix/random_number_source.h"


// On Posix platforms include rand_r
#include <stdlib.h>


kapaga::random_number_source::random_number_source( source_type seed )
: source_( seed )
{
	// Nothing to do.
}



kapaga::random_number_source::value_type 
kapaga::random_number_source::draw() 
{
	return rand_r( &source_ );
}

kapaga::random_number_source::source_type 
kapaga::random_number_source::source() const 
{
	return source_;
}


void 
kapaga::random_number_source::set_source( source_type source ) 
{
	source_ = source;
}


kapaga::random_number_source::source_type
kapaga::random_number_source::default_seed()
{
	return static_cast< source_type >( default_seed_value );
}


kapaga::random_number_source::value_type 
kapaga::random_number_source::min() 
{
	return static_cast< value_type >( 0 );
}

kapaga::random_number_source::value_type 
kapaga::random_number_source::max() 
{
	return static_cast< value_type >( RAND_MAX );
}
