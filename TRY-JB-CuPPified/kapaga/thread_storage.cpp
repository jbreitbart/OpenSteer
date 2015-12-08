/**
 * Kapaga: Kassel Parallel Games
 *
 * Copyright (c) 2006-2007, Kapaga Development Group
 * All rights reserved.
 *
 * This file is part of the Kapaga project.
 * For conditions of distribution and use, see copyright notice in kapaga_license.txt.
 */

#include "kapaga/thread_storage.h"


kapaga::size_t 
kapaga::max_needed_thread_slot_count()
{
	return omp_get_max_threads();
}


kapaga::size_t 
kapaga::needed_thread_slot_count()
{
	return omp_get_num_threads();
}
