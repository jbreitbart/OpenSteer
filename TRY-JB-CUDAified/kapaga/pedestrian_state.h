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
 * Everything that is needed to simulate a pedestrian
 */

///@note no namespace as we need this in cuda too

#ifndef KAPAGA_pedestrian_state_H
#define KAPAGA_pedestrian_state_H

#include <vector_types.h> // CUDA vector types


///@warning Alignment needed or transfer gcc compiled -> nvcc compiled CPU will not work(!)
///@warning Struct internal alignment is not supported, see http://forums.nvidia.com/index.php?showtopic=28930&hl=struct
///@todo Ask how structs can be passed from one compiler to another

typedef struct
{
	// the data needed for the agents
	float3 vehicle_steeringForces __attribute__((aligned(16)));
	float3 vehicle_forward __attribute__((aligned(16)));
	float3 vehicle_position __attribute__((aligned(16)));
	float3 vehicle_side __attribute__((aligned(16)));
	float4 vehicle_wander_rand __attribute__((aligned(16)));
	float3 vehicle_up __attribute__((aligned(16)));
	float2 vehicle_random __attribute__((aligned(16)));
	float  vehicle_speed __attribute__((aligned(8)));
	float  vehicle_max_force __attribute__((aligned(8)));
	float  vehicle_radius __attribute__((aligned(8)));
	float  vehicle_max_speed __attribute__((aligned(8)));
	int    vehicle_pathDirection __attribute__((aligned(8)));
}__attribute__((aligned(16))) pedestrian_state;

#endif
