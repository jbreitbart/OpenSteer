/// @todo missing header

// includes, system
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <cuda.h>

#include "float3_math.h"

/// @todo check all floating point constants to be real floats
/// @todo check all functions to call the floating point version


/// @warning GPU memory is allocated even if the plugin is not active

const unsigned int BLOCK_SIZE=64;


__device__ float3 globalizeDirection (
                     const float3 localDirection,
                     const float3 side,
                     const float3 up,
                     const float3 forward
                  )
{
	return ((side    * localDirection.x) +
	        (up      * localDirection.y) +
	        (forward * localDirection.z)
	       );
};

__device__ float3 localizeDirection (
                     const float3 globalDirection,
                     const float3 side,
                     const float3 up,
                     const float3 forward
                  )
{
	// dot offset with local basis vectors to obtain local coordiantes
	return make_float3 (dot (globalDirection, side),
	                    dot (globalDirection, up),
	                    dot (globalDirection, forward)
	                   );
};

// transform a point in global space to its equivalent in local space
__device__ float3 localizePosition (
                     const float3 globalPosition,
                     const float3 position,
                     const float3 side,
                     const float3 up,
                     const float3 forward
                  )
{
	// dot offset with local basis vectors to obtain local coordiantes
	return localizeDirection (globalPosition - position, side, up, forward);
};

const int seen_from_outside = -1;
const int seen_from_both    =  0;
const int seen_from_inside  =  1;

__device__ bool xyPointInsideShape (const float3 point, const float radius, const float width, const float height)
{
	const float w = radius + (width  * 0.5f);
	const float h = radius + (height * 0.5f);
	return !((point.x >  w) || (point.x < -w) || (point.y >  h) || (point.y < -h));
}


__device__ void  find_intersection_with_vehicle_path_and_rectangle_object (
                    // object data
                    const float3 object_position,
                    const float3 object_side,
                    const float3 object_up,
                    const float3 object_forward,
                    const int    object_seen_from,
                    const float  object_width,
                    const float  object_height,

                    // vehicle data
                    const float3 vehicle_forward,
                    const float3 vehicle_position,
                    const float  vehicle_radius,

                   // return
                   float3 &result_steer_hint,
                   bool   &result_intersect,
                   float  &result_distance
                 )
{
	// initialize pathIntersection object to "no intersection found"
	result_intersect = false;

	const float3 lp =  localizePosition (vehicle_position, object_position, object_side, object_up, object_forward);
	const float3 ld = localizeDirection (vehicle_forward, object_side, object_up, object_forward);

	// no obstacle intersection if path is parallel to XY (side/up) plane
	if (dot (ld, make_float3 (0.0f, 0.0f, 1.0f)) == 0.0f) return;

	// no obstacle intersection if vehicle is heading away from the XY plane
	if ((lp.z > 0.0f) && (ld.z > 0.0f)) return;
	if ((lp.z < 0.0f) && (ld.z < 0.0f)) return;

	// no obstacle intersection if obstacle "not seen" from vehicle's side
	if ((object_seen_from == seen_from_outside) && (lp.z < 0.0f)) return;
	if ((object_seen_from == seen_from_inside)  && (lp.z > 0.0f)) return;

	// find intersection of path with rectangle's plane (XY plane)
	const float ix = lp.x - (ld.x * lp.z / ld.z);
	const float iy = lp.y - (ld.y * lp.z / ld.z);
	const float3 planeIntersection = make_float3 (ix, iy, 0.0f);

	// no obstacle intersection if plane intersection is outside 2d shape
	if (!xyPointInsideShape (planeIntersection, vehicle_radius, object_width, object_height)) return;

	// otherwise, the vehicle path DOES intersect this rectangle
	const float3 localXYradial = normalize(planeIntersection);
	const float3 radial        = globalizeDirection (localXYradial, object_side, object_up, object_forward);
	const float  sideSign       = (lp.z > 0.0f) ? +1.0f : -1.0f;
	const float3 opposingNormal  = object_forward * sideSign;
	
	result_intersect           = true;
	result_distance            = length (lp - planeIntersection);
	result_steer_hint          = opposingNormal + radial; // should have "toward edge" term?

#if 0
	pi.surfacePoint = globalizePosition (planeIntersection);
	pi.surfaceNormal = opposingNormal;
	pi.vehicleOutside = lp.z > 0.0f;
	pi.obstacle = this;
#endif
}

__device__ void find_intersection_with_vehicle_path_and_sphere_object (
                    // object data
                    const float3 object_center,
                    const float  object_radius,
                    const int    object_seen_from,
                    

                    // vehicle data
                    const float3 vehicle_forward,
                    const float3 vehicle_position,
                    const float vehicle_radius,
                    const float3 vehicle_side,
                    const float3 vehicle_up,
                    
                   // return
                   float3 &result_steer_hint,
                   bool   &result_intersect,
                   float  &result_distance
                )
{
	// This routine is based on the Paul Bourke's derivation in:
	//   Intersection of a Line and a Sphere (or circle)
	//   http://www.swin.edu.au/astronomy/pbourke/geometry/sphereline/
	// But the computation is done in the vehicle's local space, so
	// the line in question is the Z (Forward) axis of the space which
	// simplifies some of the calculations.

	// initialize pathIntersection object to "no intersection found"
	result_intersect = false;

	// find sphere's "local center" (lc) in the vehicle's coordinate space
	const float3 lc = localizePosition (object_center, vehicle_position, vehicle_side, vehicle_up, vehicle_forward);

	// compute line-sphere intersection parameters
	const float r = object_radius + vehicle_radius;
	const float b = -2 * lc.z;
	
	const float c = length_squared(lc) - r*r;
	
	const float d = (b * b) - (4 * c);

	// when the path does not intersect the sphere
	if (d < 0) return;

	// otherwise, the path intersects the sphere in two points with
	// parametric coordinates of "p" and "q".  (If "d" is zero the two
	// points are coincident, the path is tangent)
	const float s = sqrtf (d);
	const float p = (-b + s) / 2;
	const float q = (-b - s) / 2;

	// both intersections are behind us, so no potential collisions
	if ((p < 0) && (q < 0)) return;

	// at least one intersection is in front, so intersects our forward
	// path
	result_intersect = true;
	result_distance =
	                  ((p > 0) && (q > 0)) ?
	                  // both intersections are in front of us, find nearest one
	                  ((p < q) ? p : q) :
	                  // otherwise one is ahead and one is behind: we are INSIDE obstacle
	                  (object_seen_from == seen_from_outside ?
	                  // inside a solid obstacle, so distance to obstacle is zero
	                  0.0f :
	                  // hollow obstacle (or "both"), pick point that is in front
	                  ((p > 0) ? p : q));
	
	// hmm, note that this was actually determined already in pi.distance calc
	const bool vehicleOutside  = length (lc) > object_radius;
	const float3 surfacePoint  = vehicle_position + (vehicle_forward * result_distance);
	const float3 surfaceNormal = normalize(surfacePoint-object_center);
	
	switch (object_seen_from)
	{
		case seen_from_outside:
			result_steer_hint = surfaceNormal;
			break;
		case seen_from_inside:
			result_steer_hint = -surfaceNormal;
			break;
		case seen_from_both:
			result_steer_hint = surfaceNormal * (vehicleOutside ? 1.0f : -1.0f);
			break;
	}
#if 0
	pi.obstacle = this;
#endif
}

__device__ void firstPathIntersectionWithObstacleGroup (
                   //data needed for rectangle objects
                   const unsigned int number_of_rectangle_objects,
                   const float3* object_rectangle_position,
                   const float3* object_rectangle_side,
                   const float3* object_rectangle_up,
                   const float3* object_rectangle_forward,
                   const int*    object_rectangle_seen_from,
                   const float*  object_rectangle_width,
                   const float*  object_rectangle_height,

                   //data needed for sphere objects
                   const unsigned int number_of_sphere_objects,
                   const float3* object_sphere_center,
                   const float*  object_sphere_radius,
                   const int*    object_sphere_seen_from,

                   // vehicle data
                   const float3 vehicle_forward,
                   const float3 vehicle_position,
                   const float  vehicle_radius,
                   const float3 vehicle_side,
                   const float3 vehicle_up,

                   // return
                   float3 &result_steer_hint,
                   bool   &result_intersect,
                   float  &result_distance
                )
{
	// test all obstacles in group for an intersection with the vehicle's
	// future path, select the one whose point of intersection is nearest
	bool   next_intersect    = false;
	float  next_distance     = 0.0f;
	float3 next_steer_hint   = make_float3 (0.0f, 0.0f, 0.0f);
	       result_intersect  = false;
	       result_distance   = 0.0f;
	       result_steer_hint = make_float3 (0.0f, 0.0f, 0.0f);

	// first rectangle objects
	for (unsigned int i=0; i< number_of_rectangle_objects; ++i) {
		find_intersection_with_vehicle_path_and_rectangle_object (
		   // object data
		   object_rectangle_position[i],
		   object_rectangle_side[i],
		   object_rectangle_up[i],
		   object_rectangle_forward[i],
		   object_rectangle_seen_from[i],
		   object_rectangle_width[i],
		   object_rectangle_height[i],

		   // vehicle data
		   vehicle_forward,
		   vehicle_position,
		   vehicle_radius,

		   // return
		   next_steer_hint,
		   next_intersect,
		   next_distance
		);

		// if this is the first intersection found, or it is the nearest found
		// so far, store it in PathIntersection object "nearest"
		const bool first_found   = !result_intersect;
		const bool nearest_found = (next_intersect && (next_distance < result_distance));
		if (first_found || nearest_found) {
			result_steer_hint = next_steer_hint;
			result_intersect  = next_intersect;
			result_distance   = next_distance;
		}
	}

	// second sphere objects
	for (unsigned int i=0; i< number_of_sphere_objects; ++i) {
		find_intersection_with_vehicle_path_and_sphere_object (
		   // object data
		   object_sphere_center[i],
		   object_sphere_radius[i],
		   object_sphere_seen_from[i],

		   // vehicle data
		   vehicle_forward,
		   vehicle_position,
		   vehicle_radius,
		   vehicle_side,
		   vehicle_up,

		   // return
		   next_steer_hint,
		   next_intersect,
		   next_distance
		);

		// if this is the first intersection found, or it is the nearest found
		// so far, store it in PathIntersection object "nearest"
		const bool first_found   = !result_intersect;
		const bool nearest_found = (next_intersect && (next_distance < result_distance));
		if (first_found || nearest_found) {
			result_steer_hint = next_steer_hint;
			result_intersect  = next_intersect;
			result_distance   = next_distance;
		}
	}
}

__device__ float3 steerToAvoidIfNeeded (
                     const float  vehicle_speed,
                     const float3 vehicle_forward,
                     const float  vehicle_max_force,
                     const float3 path_intersection_steer_hint,
                     const bool   path_intersection_intersect,
                     const float  path_intersection_distance,
                     const float  min_time_to_collision
                  )
{
	// if nearby intersection found, steer away from it, otherwise no steering
	const float minDistanceToCollision = min_time_to_collision * vehicle_speed;
	
	if (path_intersection_intersect && (path_intersection_distance < minDistanceToCollision)) {
		// compute avoidance steering force: take the component of
		// steerHint which is lateral (perpendicular to vehicle's
		// forward direction), set its length to vehicle's maxForce
		const float3 lateral = perpendicular_component (path_intersection_steer_hint, vehicle_forward);
		return normalize(lateral) * vehicle_max_force;
	} else {
		return make_float3 (0.0f, 0.0f, 0.0f);
	}
}


__device__ float3 steerToAvoidObstacles (
                       const float min_time_to_collision,
                       
                       //data needed for rectangle objects
                       const unsigned int number_of_rectangle_objects,
                       const float3* object_rectangle_position,
                       const float3* object_rectangle_side,
                       const float3* object_rectangle_up,
                       const float3* object_rectangle_forward,
                       const int*    object_rectangle_seen_from,
                       const float*  object_rectangle_width,
                       const float*  object_rectangle_height,

                       //data needed for sphere objects
                       const unsigned int number_of_sphere_objects,
                       const float3* object_sphere_center,
                       const float*  object_sphere_radius,
                       const int*    object_sphere_seen_from,

                       // vehicle data
                       const float3 vehicle_forward,
                       const float3 vehicle_position,
                       const float  vehicle_radius,
                       const float3 vehicle_side,
                       const float3 vehicle_up,
                       const float  vehicle_speed,
                       const float  vehicle_max_force
                   )
{
	float3 path_intersection_steer_hint;
	bool   path_intersection_intersect;
	float  path_intersection_distance;

	// test all obstacles in group for an intersection with the vehicle's
	// future path, select the one whose point of intersection is nearest
	firstPathIntersectionWithObstacleGroup (
	   //data needed for rectangle objects
	   number_of_rectangle_objects,
	   object_rectangle_position,
	   object_rectangle_side,
	   object_rectangle_up,
	   object_rectangle_forward,
	   object_rectangle_seen_from,
	   object_rectangle_width,
	   object_rectangle_height,

	   //data needed for sphere objects
	   number_of_sphere_objects,
	   object_sphere_center,
	   object_sphere_radius,
	   object_sphere_seen_from,

	   // vehicle data
	   vehicle_forward,
	   vehicle_position,
	   vehicle_radius,
	   vehicle_side,
	   vehicle_up,

	   // return
	   path_intersection_steer_hint,
	   path_intersection_intersect,
	   path_intersection_distance
	);

	// if nearby intersection found, steer away from it, otherwise no steering
	return steerToAvoidIfNeeded (
	          vehicle_speed,
	          vehicle_forward,
	          vehicle_max_force,
	          path_intersection_steer_hint,
	          path_intersection_intersect,
	          path_intersection_distance,
	          min_time_to_collision
	       );
}

#if 0
__device__ float3 steerToAvoidObstacles (const float minTimeToCollision, const ObstacleGroup& obstacles)
{
	///@todo
	//if (avoidance != Vec3::zero)
	//	annotateAvoidObstacle (minTimeToCollision * speed());

	return steerToAvoidObstacles (*this, minTimeToCollision, obstacles);
}
#endif


__device__ float3 steerToAvoidCloseNeighbors (
                     const float min_separation_distance,
                     const unsigned int crowd_size,
                     const float   my_radius,
                     const float*  neighbor_radius,
                     const float3  my_position,
                     const float3* neighbor_position,
                     const float3  my_forward
                  )
{
	// for each of the other vehicles...
	for (unsigned int i=0; i<crowd_size; ++i) {
		const float  sum_of_radii = my_radius + neighbor_radius[i];
		const float  min_center_to_center = min_separation_distance + sum_of_radii;
		const float3 offset = neighbor_position[i] - my_position;
		const float  current_distance = length(offset);

		if (current_distance < min_center_to_center)
		{
			//annotateAvoidCloseNeighbor (other, minSeparationDistance);
			return perpendicular_component(-offset, my_forward);
		}
	}
	// otherwise return zero
	return make_float3(0.0f, 0.0f, 0.0f);
}


__device__ float predictNearestApproachTime (
                    const float3 my_velocity,
                    const float3 other_velocity,
                    const float3 my_position,
                    const float3 other_position
                 )
{
	// imagine we are at the origin with no velocity,
	// compute the relative velocity of the other vehicle
	const float3 rel_velocity = other_velocity - my_velocity;
	const float  rel_speed    = length(rel_velocity);

	// for parallel paths, the vehicles will always be at the same distance,
	// so return 0 (aka "now") since "there is no time like the present"
	if (rel_speed == 0.0f) return 0.0f;

	// Now consider the path of the other vehicle in this relative
	// space, a line defined by the relative position and velocity.
	// The distance from the origin (our vehicle) to that line is
	// the nearest approach.

	// Take the unit tangent along the other vehicle's path
	const float3 rel_tangent = rel_velocity / rel_speed;

	// find distance from its path to origin (compute offset from
	// other to us, find length of projection onto path)
	const float3 rel_position = my_position - other_position;
	const float  projection   = dot(rel_tangent, rel_position);

	return projection / rel_speed;
}


__device__ float computeNearestApproachPositions (
                    float time,
                    const float3 my_forward,
                    const float my_speed,
                    const float3 my_position,
                    const float3 other_forward,
                    const float other_speed,
                    const float3 other_position,
//                    Vec3& ourPositionAtNearestApproach,
                    float3& hisPositionAtNearestApproach
                 )
{
	const float3 my_travel    = my_forward    * my_speed    * time;
	const float3 other_travel = other_forward * other_speed * time;

	const float3 my_final    = my_position    + my_travel;
	const float3 other_final = other_position + other_travel;

	// xxx for annotation
	//ourPositionAtNearestApproach = myFinal;
	hisPositionAtNearestApproach = other_final;

	return distance(my_final, other_final);
}


__device__ float3 steerToAvoidNeighbors (
                     const float minTimeToCollision,
                     const unsigned int crowd_size,
                     const float   my_radius,
                     const float3  my_position,
                     const float3  my_forward,
                     const float3  my_velocity,
                     const float   my_speed,
                     const float3  my_side,
                     const float * neighbor_radius,
                     const float3* neighbor_position,
                     const float3* neighbor_velocity,
                     const float3* neighbor_forward,
                     const float * neighbor_speed
                 )
{
	// first priority is to prevent immediate interpenetration
	float3 separation = steerToAvoidCloseNeighbors(
	                             0.0f,
	                             crowd_size,
	                             my_radius,
	                             neighbor_radius,
	                             my_position,
	                             neighbor_position,
	                             my_forward
	                          );
	                          
	if (separation != make_float3(0.0f, 0.0f, 0.0f)) return separation;

	// otherwise, go on to consider potential future collisions
	float steer = 0.0f;

	// Time (in seconds) until the most immediate collision threat found
	// so far.  Initial value is a threshold: don't look more than this
	// many frames into the future.
	float minTime = minTimeToCollision;

	// xxx solely for annotation
	/// @todo really?
	float3 xxxThreatPositionAtNearestApproach = make_float3(0.0f, 0.0f, 0.0f);
	//float3 xxxOurPositionAtNearestApproach = make_float3(0.0f, 0.0f, 0.0f);
	float3 threat_forward  = make_float3 (0.0f, 0.0f, 0.0f);
	float3 threat_position = make_float3 (0.0f, 0.0f, 0.0f);
	float3 threat_velocity = make_float3 (0.0f, 0.0f, 0.0f);
	float  threat_speed    = 0.0f;


	// for each of the other vehicles, determine which
	// (if any)
	// pose the most immediate threat of collision.
	// bknafla: One possible way to parallelize this (though the improvements through nested
	//          parallelism are questionable) is to store the nearest approach distance or time
	//          at the index of the different agents and search for the smalles value later on.
	//          Profile before parallelizing!
	///@todo think about this for GPGPU
	// avoid when future positions are this close (or less)
	
	const float collisionDangerThreshold = my_radius * 2.0f;
	
	for (unsigned int i=0; i<crowd_size; ++i) {
		
		// predicted time until nearest approach of "this" and "other"
		const float time = predictNearestApproachTime(
					my_velocity,
					neighbor_velocity[i],
					my_position,
					neighbor_position[i]
					);

		// If the time is in the future, sooner than any other
		// threatened collision...
		if ((time >= 0) && (time < minTime))
		{
			// if the two will be close enough to collide,
			// make a note of it
			// float3 ourPositionAtNearestApproach = make_float3(0.0f, 0.0f, 0.0f);
			// float3 hisPositionAtNearestApproach = make_float3(0.0f, 0.0f, 0.0f);
			if (computeNearestApproachPositions (
				time,
				my_forward,
				my_speed,
				my_position,
				neighbor_forward[i],
				neighbor_speed[i],
				neighbor_position[i],
				xxxThreatPositionAtNearestApproach
				) < collisionDangerThreshold)
			{
				minTime = time;
				threat_forward  = neighbor_forward[i];
				threat_position = neighbor_position[i];
				threat_velocity = neighbor_velocity[i];
				threat_speed    = neighbor_speed[i];
				//xxxThreatPositionAtNearestApproach
				//	= hisPositionAtNearestApproach;
				//xxxOurPositionAtNearestApproach
				//	= ourPositionAtNearestApproach;
			}
		}
	}
	// if a potential collision was found, compute steering to avoid
	if (threat_speed != 0.0f)
	{
		// parallel: +1, perpendicular: 0, anti-parallel: -1
		const float parallelness = dot (my_forward, threat_forward);
		const float angle = 0.707f;

		if (parallelness < -angle)
		{
			// anti-parallel "head on" paths:
			// steer away from future threat position
			const float3 offset  = xxxThreatPositionAtNearestApproach - my_position;
			const float  sideDot = dot (offset, my_side);
			steer = (sideDot > 0) ? -1.0f : 1.0f;
		} else {
			if (parallelness > angle)
			{
				// parallel paths: steer away from threat
				const float3 offset = threat_position - my_position;
				const float sideDot = dot (offset, my_side);
				steer = (sideDot > 0) ? -1.0f : 1.0f;
			}
			else
			{
				// perpendicular paths: steer behind threat
				// (only the slower of the two does this)
				if (threat_speed <= my_speed)
				{
					const float sideDot = dot (my_side, threat_velocity);
					steer = (sideDot > 0) ? -1.0f : 1.0f;
				}
			}
		}

	}

	return my_side * steer;
}

__device__ float scalarRandomWalk (
                    const float initial,
                    const float walkspeed,
                    const float min,
                    const float max,
                    const float random
                 )
{
	float next = initial + (((random * 2) - 1) * walkspeed);
	if (next < min) next=min;
	if (next > max) next=max;

	return next;
}

__device__ float3 steerForWander (
                     const float  dt,
                     const float3 my_side,
                     const float3 my_up,
                     const float bi_rand1,
                     const float bi_rand2,
                     const float rand1,
                     const float rand2
                  )
{
	// random walk WanderSide and WanderUp between -1 and +1
	const float speed = 12.0f * dt; // maybe this (12) should be an argument?
	float const wanderSidewaysUrge = scalarRandomWalk (bi_rand1, speed, -1, +1, rand1 );
	float const wanderUpUrge       = scalarRandomWalk (bi_rand2, speed, -1, +1, rand2 );

	// return a pure lateral steering vector: (+/-Side) + (+/-Up)
	return (my_side * wanderSidewaysUrge) + (my_up * wanderUpUrge );
}


__device__ float3 predictFuturePosition (
                     const float  predictionTime,
                     const float3 my_position,
                     const float3 my_velocity
                  )
{
	return my_position + (my_velocity * predictionTime);
}


__device__ void mapPointToSegmentDistanceAndPointAndTangentAndRadius(
                    const unsigned int  segmentIndex,
                    const float3        point,
                    const float3* const path_points,
                    const float3* const path_segmentTangents,
                    const float * const path_segmentLengths,
                    const float         path_radius,
                          float &       distance,
                          float3&       pointOnPath,
                          float3&       tangent,
                          float &       radius
                )
{
	const float3 segmentStartPoint = path_points[ segmentIndex ];
	const float3 segmentStartToPoint = ( point - segmentStartPoint );
	tangent = path_segmentTangents[ segmentIndex ];
	distance = dot (segmentStartToPoint, tangent);
	if (distance<0.0f) distance = 0.0f;
	if (distance>path_segmentLengths[segmentIndex]) distance = path_segmentLengths[segmentIndex];
	pointOnPath = tangent * distance + segmentStartPoint;
	radius = path_radius;
}


__device__ void mapPointToPathAlike_extract(
                   const unsigned int  segmentIndex,
                   const float3        point,
                   const float3* const path_points,
                   const float3* const path_segmentTangents,
                   const float * const path_segmentLengths,
                   const float         path_radius,
                         float &       segmentDistance,
                         float &       radius,
                         float &       distancePointToPath,
                         float3&       pointOnPathCenterLine,
                         float3&       tangent
                )
{
	mapPointToSegmentDistanceAndPointAndTangentAndRadius(
	   segmentIndex,
	   point,
	   path_points,
	   path_segmentTangents,
	   path_segmentLengths,
	   path_radius,
	   segmentDistance,
	   pointOnPathCenterLine,
	   tangent,
	   radius
	);
	distancePointToPath = distance( point, pointOnPathCenterLine ) - radius;
}


__device__ void  mapPointToPathAlike(
                    const float3        queryPoint,
                    const unsigned int  path_segment_count,
                    const float3* const path_points,
                    const float3* const path_segmentTangents,
                    const float * const path_segmentLengths,
                    const float         path_radius,

                    //return
                    float  &       DistanceOnPath,
                    float3 &       tangent,
                    float  &       distancePointToPath,
                    float3 &       pointOnPathCenterLine
                 )
{
	float minDistancePointToPath     = 9E10;
	float DistanceOnPathFlag         = 0.0f;

	for ( unsigned int i=0; i<path_segment_count; ++i) {

		float segmentDistance = 0.0f;
		float radius = 0.0f;
		float temp_distancePointToPath = 0.0f;
		
		float3 temp_pointOnPathCenterLine = make_float3( 0.0f, 0.0f, 0.0f );
		float3 temp_tangent = make_float3( 0.0f, 0.0f, 0.0f );

		mapPointToPathAlike_extract(
		   i,
		   queryPoint,
		   path_points,
                   path_segmentTangents,
                   path_segmentLengths,
                   path_radius,
		   segmentDistance,
		   radius,
		   temp_distancePointToPath,
		   temp_pointOnPathCenterLine,
		   temp_tangent
		);

		if ( temp_distancePointToPath < minDistancePointToPath ) {
			minDistancePointToPath = temp_distancePointToPath;
#if 0
			mapping.setPointOnPathCenterLine( pointOnPathCenterLine );
			mapping.setPointOnPathBoundary( pointOnPathCenterLine+(( queryPoint - pointOnPathCenterLine ).normalize() * radius ));
			mapping.setRadius( radius );
			mapping.setTangent( tangent );
			mapping.setSegmentIndex( segmentIndex );
			mapping.setDistancePointToPath( distancePointToPath );
			mapping.setDistancePointToPathCenterLine( distancePointToPath + radius );
			mapping.setDistanceOnSegment( segmentDistance );
#endif
			DistanceOnPath        = DistanceOnPathFlag + segmentDistance;
			tangent               = temp_tangent;
			distancePointToPath   = temp_distancePointToPath;
			pointOnPathCenterLine = temp_pointOnPathCenterLine; 
		}

		DistanceOnPathFlag = DistanceOnPathFlag + path_segmentLengths[i];
	}

}


__device__ float mapPointToPathDistance (
                    const float3        queryPoint,
                    const unsigned int  path_segment_count,
                    const float3* const path_points,
                    const float3* const path_segmentTangents,
                    const float * const path_segmentLengths,
                    const float         path_radius
                 )
{
	float  DistanceOnPath;
	float3 tangent;
	float distancePointToPath;
	float3 pointOnPathCenterLine;
	mapPointToPathAlike(
	           queryPoint,
	           path_segment_count,
	           path_points,
	           path_segmentTangents,
	           path_segmentLengths,
	           path_radius,
                   
                   //return
	           DistanceOnPath,
	           tangent,
	           distancePointToPath,
	           pointOnPathCenterLine
	       );
	return distancePointToPath;
}

__device__ void mapDistanceToSegmentPointAndTangent (
                   const unsigned int         segmentIndex,
                   const float        * const path_segmentLengths,
                   const float3       * const path_segmentTangents,
                   const float3       * const path_points,
                         float                segmentDistance,
                         float3       &       pointOnPath,
                         float3       &       tangent
                )
{
	const float segmentLength = path_segmentLengths[ segmentIndex ];

	if (segmentDistance<0.0f) segmentDistance = 0.0f;
	if (segmentDistance>segmentLength) segmentDistance = segmentLength;

	pointOnPath = path_segmentTangents[ segmentIndex ] * segmentDistance + path_points[ segmentIndex ];
	tangent = path_segmentTangents[ segmentIndex ];
}


__device__ void mapDistanceToSegmentPointAndTangentAndRadius(
                   const unsigned int         segmentIndex,
                   const float        * const path_segmentLengths,
                   const float3       * const path_segmentTangents,
                   const float3       * const path_points,
                   const float                path_radius,
                   const float                distance,
                         float3       &       pointOnPath,
                         float3       &       tangent,
                         float        &       radius
                )
{
	mapDistanceToSegmentPointAndTangent(
	   segmentIndex,
	   path_segmentLengths,
	   path_segmentTangents,
	   path_points,
	   distance,
	   pointOnPath,
	   tangent
	);
	radius = path_radius;
}


__device__ void mapDistanceToPathAlike_extract(
                   const unsigned int         segmentIndex,
                   const float        * const path_segmentLengths,
                   const float3       * const path_segmentTangents,
                   const float3       * const path_points,
                   const float                SegmentDistance,
                         float3       &       pointOnPath,
                         float3       &       tangent
)
{
	mapDistanceToSegmentPointAndTangent(
	  segmentIndex,
	  path_segmentLengths,
	  path_segmentTangents,
	  path_points,
	  SegmentDistance,
	  pointOnPath,
	  tangent
	);
}

__device__ float3 mapDistanceToPathAlike(
                     const float                path_length,
                     const bool                 path_is_cyclic,
                     const unsigned int         path_segment_count,
                     const float3*        const path_points,
                     const float3*        const path_segmentTangents,
                     const float *        const path_segmentLengths,
                     const float                path_radius,
                     float                      distanceOnPath
                  )
{

	float const pathLength = path_length;

	// Modify @c distanceOnPath to applicable values.
	if ( path_is_cyclic ) {
		distanceOnPath = fmod(distanceOnPath, pathLength);
	}

	if (distanceOnPath<0.0f) distanceOnPath = 0.0f;
	if (distanceOnPath>pathLength) distanceOnPath = pathLength;

	// Which path alike segment is reached by @c distanceOnPath?
	float              remainingDistance = distanceOnPath;
	unsigned int       segmentIndex      = 0;
	const unsigned int maxSegmentIndex   = path_segment_count - 1;
	
	while( ( segmentIndex < maxSegmentIndex ) &&
		( remainingDistance > path_segmentLengths[segmentIndex] ) ) {
		remainingDistance -= path_segmentLengths[segmentIndex];
		++segmentIndex;
	}

	// Extract the path related data associated with the segment reached
	// by @c distanceOnPath.
	float3 pointOnPathCenterLine = make_float3( 0.0f, 0.0f, 0.0f );
	float3 tangent               = make_float3( 0.0f, 0.0f, 0.0f );
	mapDistanceToPathAlike_extract(
	   segmentIndex,
	   path_segmentLengths,
	   path_segmentTangents,
	   path_points,
	   remainingDistance,
	   pointOnPathCenterLine,
	   tangent
	);

	return pointOnPathCenterLine;
	
	// Store the extracted data in @c mapping to return it to the caller.
#if 0
	mapping.setRadius( radius );
	mapping.setTangent( tangent );
	mapping.setSegmentIndex( segmentIndex );
	mapping.setDistanceOnPath( distanceOnPath );
	mapping.setDistanceOnSegment( remainingDistance );
#endif
}



__device__ float3 steerForSeek (const float3 target, const float3 my_position, const float3 my_velocity)
{
	const float3 desiredVelocity = target - my_position;
	return desiredVelocity - my_velocity;
}


__device__ float3 mapPointToPath (
                    const float3        queryPoint,
                    const unsigned int  path_segment_count,
                    const float3* const path_points,
                    const float3* const path_segmentTangents,
                    const float * const path_segmentLengths,
                    const float         path_radius,

                    // return
                           float3 &     tangent,
                           float  &     outside
                  )
{
	float   DistanceOnPath;
	float   distancePointToPath;
	float3  pointOnPathCenterLine;
	mapPointToPathAlike(
	           queryPoint,
	           path_segment_count,
	           path_points,
	           path_segmentTangents,
	           path_segmentLengths,
	           path_radius,
                   
                   //return
	           DistanceOnPath,
	           tangent,
	           distancePointToPath,
	           pointOnPathCenterLine
	       );
	
	outside = distancePointToPath;
	return pointOnPathCenterLine;
}

__device__ float3 steerToFollowPath (
                     const int           direction,
                     const float         predictionTime,
                     const float         my_speed,
                     const float3        my_position,
                     const float3        my_velocity,

                     const unsigned int  path_segment_count,
                     const float3* const path_points,
                     const float3* const path_segmentTangents,
                     const float * const path_segmentLengths,
                     const float         path_radius,
                     const float         path_length,
                     const bool          path_is_cyclic
                  )
{
	// our goal will be offset from our path distance by this amount
	const float pathDistanceOffset = direction * predictionTime * my_speed;

	// predict our future position
	const float3 futurePosition = predictFuturePosition (
	                               predictionTime,
	                               my_position,
	                               my_velocity
	                            );

	// measure distance along path of our current and predicted positions
	float3 tangent;
	float distancePointToPath;
	float3 pointOnPathCenterLine;
	
	float nowPathDistance;
	mapPointToPathAlike (
	   my_position,
	   path_segment_count,
	   path_points,
	   path_segmentTangents,
	   path_segmentLengths,
	   path_radius,
	   
	   //return
	   nowPathDistance,
	   tangent,
	   distancePointToPath,
	   pointOnPathCenterLine
	);
	
	float futurePathDistance;
	mapPointToPathAlike (
	   futurePosition,
	   path_segment_count,
	   path_points,
	   path_segmentTangents,
	   path_segmentLengths,
	   path_radius,
	   
	   //return
	   futurePathDistance,
	   tangent,
	   distancePointToPath,
	   pointOnPathCenterLine
	);

	// bknafla: @todo Put this calculation into its own function and call this function in the
	//                if-test further down (lazy evaluation).
	// are we facing in the correction direction?
	const bool rightway = ((pathDistanceOffset > 0) ?
				(nowPathDistance < futurePathDistance) :
				(nowPathDistance > futurePathDistance));
				
	tangent = make_float3( 0.0f, 0.0f, 0.0f);
	float outside = 0.0f;
	const float3 onPath = mapPointToPath (
	                         futurePosition,
	                         path_segment_count,
	                         path_points,
	                         path_segmentTangents,
	                         path_segmentLengths,
	                         path_radius,
	                         tangent,
	                         outside
	                      );

	// no steering is required if (a) our future position is inside
	// the path tube and (b) we are facing in the correct direction
	if ((outside < 0) && rightway)
	{
		// all is well, return zero steering
		return make_float3(0.0f, 0.0f, 0.0f);
	} else {
		// otherwise we need to steer towards a target point obtained
		// by adding pathDistanceOffset to our current path position

		const float  targetPathDistance = nowPathDistance + pathDistanceOffset;
		const float3 target = mapDistanceToPathAlike (
		                         path_length,
		                         path_is_cyclic,
		                         path_segment_count,
		                         path_points,
		                         path_segmentTangents,
		                         path_segmentLengths,
		                         path_radius,
		                         targetPathDistance
		                      );

		// return steering to seek target on path
		return steerForSeek (target, my_position, my_velocity);
	}
}

__device__ float3 steerToStayOnPath (
                     const float predictionTime,
                     const float3 my_position,
                     const float3 my_velocity,
                     const unsigned int  path_segment_count,
                     const float3* const path_points,
                     const float3* const path_segmentTangents,
                     const float * const path_segmentLengths,
                     const float         path_radius
                  )
{
	// predict our future position
	const float3 futurePosition = predictFuturePosition (
	                               predictionTime,
	                               my_position,
	                               my_velocity
	                            );

	// find the point on the path nearest the predicted future position
	float3 tangent;
	float outside;
	const float3 onPath = mapPointToPath (
	                         futurePosition,
	                         path_segment_count,
	                         path_points,
	                         path_segmentTangents,
	                         path_segmentLengths,
	                         path_radius,
	                         tangent,
	                         outside
	                      );

	if (outside < 0)
	{
		// our predicted future position was in the path,
		// return zero steering.
		return make_float3(0.0f, 0.0f, 0.0f);
	}
	else
	{
		return steerForSeek (onPath, my_position, my_velocity);
	}
}
__global__ void determineCombinedSteering_kernel_op (
                   const float elapsedTime,
                   const unsigned int crowd_size,

                   // vehicle data
                   const float3* const vehicle_forward,
                   const float3* const vehicle_position,
                   const float2* const vehicle_random,
                   const float * const vehicle_radius,
                   const float3* const vehicle_side,
                   const float3* const vehicle_up,
                   const float*  const vehicle_speed,
                   const float*  const vehicle_max_force,
                   const float*  const vehicle_max_speed,
                   const int   * const vehicle_pathDirection,

                   // data needed for rectangle objects
                   const unsigned int number_of_rectangle_objects,
                   const float3* const object_rectangle_position,
                   const float3* const object_rectangle_side,
                   const float3* const object_rectangle_up,
                   const float3* const object_rectangle_forward,
                   const int*    const object_rectangle_seen_from,
                   const float*  const object_rectangle_width,
                   const float*  const object_rectangle_height,

                   // data needed for sphere objects
                   const unsigned int number_of_sphere_objects,
                   const float3* const object_sphere_center,
                   const float*  const object_sphere_radius,
                   const int*    const object_sphere_seen_from,

                   // wandering
                   const bool          gWanderSwitch,
                   const float4* const vehicle_wander_rand,

                   // path
                   const bool          gUseDirectedPathFollowing,
                   const unsigned int  path_segment_count,
                   const float3* const path_points,
                   const float3* const path_segmentTangents,
                   const float * const path_segmentLengths,
                   const float         path_radius,
                   const float         path_length,
                   const bool          path_is_cyclic,

                   // result
                   float3* const result,

                   // dirty hack
                   float * device_neighbor_radius,
                   float3* device_neighbor_position,
                   float3* device_neighbor_velocity,
                   float3* device_neighbor_forward,
                   float * device_neighbor_speed
	)
{
	// I don't like the cuda names ;-)
	const dim3 &thread_id = threadIdx;
	const dim3 &block_id  = blockIdx;

	// the index to the data in the global data fields
	const unsigned int index = BLOCK_SIZE * block_id.x + thread_id.x;

	// *** determineCombinedSteering *** //
	
	// move forward
	float3 steeringForce = vehicle_forward[index];

	// probability that a lower priority behavior will be given a
	// chance to "drive" even if a higher priority behavior might
	// otherwise be triggered.
	float const random0                    = vehicle_random[index].x;
	float const random1                    = vehicle_random[index].y;
	const float3 local_vehicle_forward     = vehicle_forward[index];
	const float3 local_vehicle_position    = vehicle_position[index];
	const float  local_vehicle_radius      = vehicle_radius[index];
	const float3 local_vehicle_side        = vehicle_side[index];
	const float3 local_vehicle_up          = vehicle_up[index];
	const float  local_vehicle_speed       = vehicle_speed[index];
	const float  local_vehicle_max_force   = vehicle_max_force[index];
	const float  local_vehicle_max_speed   = vehicle_max_speed[index];
	const float3 local_vehicle_velocity    = local_vehicle_forward * local_vehicle_speed;
	const float4 local_vehicle_wander_rand = vehicle_wander_rand[index];
	const int    local_vehicle_pathDirection = vehicle_pathDirection[index];
	
	float const leakThrough = 0.1f;
	
	// determine if obstacle avoidance is required
	float3 obstacleAvoidance = make_float3 (0.0f, 0.0f, 0.0f);
	if (leakThrough < random0 )
	{
		const float oTime = 6.0f; // minTimeToCollision = 6 seconds
		obstacleAvoidance = steerToAvoidObstacles (
		       oTime,

		       //data needed for rectangle objects
		       number_of_rectangle_objects,
		       object_rectangle_position,
		       object_rectangle_side,
		       object_rectangle_up,
		       object_rectangle_forward,
		       object_rectangle_seen_from,
		       object_rectangle_width,
		       object_rectangle_height,

		       //data needed for sphere objects
		       number_of_sphere_objects,
		       object_sphere_center,
		       object_sphere_radius,
		       object_sphere_seen_from,

		       // vehicle data
		       local_vehicle_forward,
		       local_vehicle_position,
		       local_vehicle_radius,
		       local_vehicle_side,
		       local_vehicle_up,
		       local_vehicle_speed,
		       local_vehicle_max_force
		);
	}

	// if obstacle avoidance is needed, do it
	steeringForce = steeringForce + obstacleAvoidance;
	
	if (obstacleAvoidance == make_float3 (0.0f, 0.0f, 0.0f) ) {
		// otherwise consider avoiding collisions with others
		float3 collisionAvoidance = make_float3(0.0f, 0.0f, 0.0f);

		if (leakThrough < random1 ) {
			const float caLeadTime = 3.0f;

			// find all neighbors within maxRadius using proximity database
			// (radius is largest distance between vehicles traveling head-on
			// where a collision is possible within caLeadTime seconds.)
			const float max_radius = caLeadTime * local_vehicle_max_speed * 2.0f;

#if 0
			__shared__ float  s_neighbor_radius[SIZE*BLOCK_SIZE];
			__shared__ float3 s_neighbor_position[SIZE*BLOCK_SIZE];
			__shared__ float3 s_neighbor_velocity[SIZE*BLOCK_SIZE];
			__shared__ float3 s_neighbor_forward[SIZE*BLOCK_SIZE];
			__shared__ float  s_neighbor_speed[SIZE*BLOCK_SIZE];
			float*  neighbor_radius = &s_neighbor_radius[thread_id.x*SIZE];
			float3* neighbor_position = &s_neighbor_position[thread_id.x*SIZE];
			float3* neighbor_velocity = &s_neighbor_velocity[thread_id.x*SIZE];
			float3* neighbor_forward = &s_neighbor_forward[thread_id.x*SIZE];
			float*  neighbor_speed = &s_neighbor_speed[thread_id.x*SIZE];
#endif
#if 0
			float  neighbor_radius[SIZE];
			float3 neighbor_position[SIZE];
			float3 neighbor_velocity[SIZE];
			float3 neighbor_forward[SIZE];
			float  neighbor_speed[SIZE];
#endif

			const int neighbor_size_max = 7;
			
			float*  neighbor_radius   = &device_neighbor_radius[index*neighbor_size_max];
			float3* neighbor_position = &device_neighbor_position[index*neighbor_size_max];
			float3* neighbor_velocity = &device_neighbor_velocity[index*neighbor_size_max];
			float3* neighbor_forward  = &device_neighbor_forward[index*neighbor_size_max];
			float*  neighbor_speed    = &device_neighbor_speed[index*neighbor_size_max];

			// @todo Remove magic variable.
			int max_neighbour_distance_index = 0;
			float max_neighbour_distance = 0.0f;
			int neighbors_size=0;
			
			float const r2 = max_radius*max_radius;
			for (int i=0; i<crowd_size; ++i) {
				if (i==index) continue;
				const float3 offset = local_vehicle_position - vehicle_position[i];
				const float d2      = length_squared(offset);
				if (d2<r2) {
					if ( neighbors_size < neighbor_size_max ) {
						if ( d2 > max_neighbour_distance ) {
							max_neighbour_distance = d2;
							max_neighbour_distance_index = neighbors_size;
						}
						neighbor_radius[neighbors_size]   = vehicle_radius[i];
						neighbor_position[neighbors_size] = vehicle_position[i];
						neighbor_speed[neighbors_size]    = vehicle_speed[i];
						neighbor_forward[neighbors_size]  = vehicle_forward[i];
						neighbor_velocity[neighbors_size] = vehicle_forward[i] * vehicle_speed[i];
						++neighbors_size;
					} else {
						
						if ( d2 < max_neighbour_distance ) {
							neighbor_radius[max_neighbour_distance_index]   = vehicle_radius[i];
							neighbor_position[max_neighbour_distance_index] = vehicle_position[i];
							neighbor_speed[max_neighbour_distance_index]    = vehicle_speed[i];
							neighbor_forward[max_neighbour_distance_index]  = vehicle_forward[i];
							neighbor_velocity[max_neighbour_distance_index] = vehicle_forward[i] * vehicle_speed[i];

							max_neighbour_distance = d2; // just temporary
							
							for ( int i = 0; i < neighbor_size_max; ++i ) {
								
								float const dist = length_squared( local_vehicle_position - neighbor_position[i] );
								if ( dist > max_neighbour_distance ) {
									max_neighbour_distance = dist;
									max_neighbour_distance_index = i;
								}
							}
						}
						
					}
					
				}
			}
			
			collisionAvoidance = steerToAvoidNeighbors (
			                        caLeadTime,
			                        neighbors_size,
			                        local_vehicle_radius,
			                        local_vehicle_position,
			                        local_vehicle_forward,
			                        local_vehicle_velocity,
			                        local_vehicle_speed,
			                        local_vehicle_side,
			                        neighbor_radius,
			                        neighbor_position,
			                        neighbor_velocity,
			                        neighbor_forward,
			                        neighbor_speed
			                     ) * 10.0f;

			// if collision avoidance is needed, do it
			// @todo Why not add it all the time? Seems to be cheaper than
			//       a conditional branch, isn't it?
			steeringForce = steeringForce + collisionAvoidance;
		}

		if (collisionAvoidance == make_float3 (0.0f, 0.0f, 0.0f)) {
			// add in wander component (according to user switch)
			if (gWanderSwitch) {
				steeringForce = steeringForce + steerForWander (
				                                   elapsedTime,
				                                   local_vehicle_side,
				                                   local_vehicle_up,
				                                   local_vehicle_wander_rand.x,
				                                   local_vehicle_wander_rand.y,
				                                   local_vehicle_wander_rand.z,
				                                   local_vehicle_wander_rand.w
				                                );
			}

			// do (interactively) selected type of path following
			const float pfLeadTime = 3.0f;
			const float3 pathFollow = (gUseDirectedPathFollowing ?
			                            steerToFollowPath (
			                             local_vehicle_pathDirection,
			                             pfLeadTime,
			                             local_vehicle_speed,
			                             local_vehicle_position,
			                             local_vehicle_velocity,
			                             path_segment_count,
			                             path_points,
			                             path_segmentTangents,
			                             path_segmentLengths,
			                             path_radius,
			                             path_length,
			                             path_is_cyclic
			                           ) :
			                           steerToStayOnPath (
			                             pfLeadTime,
			                             local_vehicle_position,
			                             local_vehicle_velocity,
			                             path_segment_count,
			                             path_points,
			                             path_segmentTangents,
			                             path_segmentLengths,
			                             path_radius
			                            ));

			steeringForce = steeringForce + pathFollow * 0.5f;
 		}
	}

	// return steering constrained to global XZ "ground" plane
	steeringForce.y = 0.0f;
	result[index] = steeringForce;
}

extern "C"
void GPU_determineCombinedSteering_op (
		const float elapsedTime,

		// vehicle data
		const unsigned int  crowd_size,
		const float3* const vehicle_forward,
		const float3* const vehicle_position,
		const float2* const vehicle_random,
		const float * const vehicle_radius,
		const float3* const vehicle_side,
		const float3* const vehicle_up,
		const float*  const vehicle_speed,
		const float*  const vehicle_max_force,
		const float*  const vehicle_max_speed,
                const int   * const vehicle_pathDirection,
                
		// data needed for rectangle objects
		const unsigned int number_of_rectangle_objects,
		const float3* const object_rectangle_position,
		const float3* const object_rectangle_side,
		const float3* const object_rectangle_up,
		const float3* const object_rectangle_forward,
		const int*    const object_rectangle_seen_from,
		const float*  const object_rectangle_width,
		const float*  const object_rectangle_height,

		// data needed for sphere objects
		const unsigned int number_of_sphere_objects,
		const float3* const object_sphere_center,
		const float*  const object_sphere_radius,
		const int*    const object_sphere_seen_from,

		// path
		const bool          gUseDirectedPathFollowing,
		const unsigned int  path_segment_count,
		const float3* const path_points,
		const float3* const path_segmentTangents,
		const float * const path_segmentLengths,
		const float         path_radius,
		const float         path_length,
		const bool          path_is_cyclic,
		
		// wander
		const bool          gWanderSwitch,
		const float4* const vehicle_wander_rand,
		
		// result
		float3* const result
	)
{
	// set the device
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "There is no device.\n");
		exit(EXIT_FAILURE);
	}
	int dev;
	for (dev = 0; dev < deviceCount; ++dev) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (strncmp(deviceProp.name, "Device Emulation", 16))
			break;
	}
	if (dev == deviceCount) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	} else cudaSetDevice(dev);

	static int currently_allocated_vehicles = 0;

	static float3* device_result = 0;
	
	// vehicle data	
	static float3* device_vehicle_forward = 0;
	static float3* device_vehicle_position = 0;
	static float2* device_vehicle_random = 0;
	static float * device_vehicle_radius = 0;
	static float3* device_vehicle_side = 0;
	static float3* device_vehicle_up = 0;
	static float * device_vehicle_speed = 0;
	static float * device_vehicle_max_force = 0;
	static float * device_vehicle_max_speed = 0;
	static float4* device_vehicle_wander_rand = 0;
	static int   * device_vehicle_pathDirection = 0;

	static float * device_neighbor_radius = 0;
	static float3* device_neighbor_position = 0;
	static float3* device_neighbor_velocity = 0;
	static float3* device_neighbor_forward = 0;
	static float * device_neighbor_speed = 0;
	
	if (currently_allocated_vehicles != crowd_size) {
		currently_allocated_vehicles = crowd_size;
		
		cudaFree (device_result);
		cudaFree (device_vehicle_forward);
		cudaFree (device_vehicle_position);
		cudaFree (device_vehicle_random);
		cudaFree (device_vehicle_radius);
		cudaFree (device_vehicle_side);
		cudaFree (device_vehicle_up);
		cudaFree (device_vehicle_speed);
		cudaFree (device_vehicle_max_force);
		cudaFree (device_vehicle_max_speed);
		cudaFree (device_vehicle_wander_rand);
		cudaFree (device_vehicle_pathDirection);
		
		cudaMalloc ((void**)(&device_result), crowd_size*sizeof(float3));
		cudaMalloc ((void**)(&device_vehicle_forward), crowd_size*sizeof(float3));
		cudaMalloc ((void**)(&device_vehicle_position), crowd_size*sizeof(float3));
		cudaMalloc ((void**)(&device_vehicle_random), crowd_size*sizeof(float2));
		cudaMalloc ((void**)(&device_vehicle_radius), crowd_size*sizeof(float));
		cudaMalloc ((void**)(&device_vehicle_side), crowd_size*sizeof(float3));
		cudaMalloc ((void**)(&device_vehicle_up), crowd_size*sizeof(float3));
		cudaMalloc ((void**)(&device_vehicle_speed), crowd_size*sizeof(float));
		cudaMalloc ((void**)(&device_vehicle_max_force), crowd_size*sizeof(float));
		cudaMalloc ((void**)(&device_vehicle_max_speed), crowd_size*sizeof(float));
		cudaMalloc ((void**)(&device_vehicle_wander_rand), crowd_size*sizeof(float4));
		cudaMalloc ((void**)(&device_vehicle_pathDirection), crowd_size*sizeof(int));


		cudaFree (device_neighbor_radius);
		cudaFree (device_neighbor_position);
		cudaFree (device_neighbor_velocity);
		cudaFree (device_neighbor_forward);
		cudaFree (device_neighbor_speed);

		cudaMalloc ((void**)(&device_neighbor_radius), 7*crowd_size*sizeof(float));
		cudaMalloc ((void**)(&device_neighbor_position), 7*crowd_size*sizeof(float3));
		cudaMalloc ((void**)(&device_neighbor_velocity), 7*crowd_size*sizeof(float3));
		cudaMalloc ((void**)(&device_neighbor_forward), 7*crowd_size*sizeof(float3));
		cudaMalloc ((void**)(&device_neighbor_speed), 7*crowd_size*sizeof(float));
	}
	
	
	cudaMemcpy (device_vehicle_forward, vehicle_forward, crowd_size*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy (device_vehicle_position, vehicle_position, crowd_size*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy (device_vehicle_random, vehicle_random, crowd_size*sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy (device_vehicle_radius, vehicle_radius, crowd_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy (device_vehicle_side, vehicle_side, crowd_size*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy (device_vehicle_up, vehicle_up, crowd_size*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy (device_vehicle_speed, vehicle_speed, crowd_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy (device_vehicle_max_force, vehicle_max_force, crowd_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy (device_vehicle_max_speed, vehicle_max_speed, crowd_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy (device_vehicle_wander_rand, vehicle_wander_rand, crowd_size*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy (device_vehicle_pathDirection, vehicle_pathDirection, crowd_size*sizeof(int), cudaMemcpyHostToDevice);


	//data needed for rectangle objects
	static float3* device_object_rectangle_position = 0;
	static float3* device_object_rectangle_side = 0;
	static float3* device_object_rectangle_up = 0;
	static float3* device_object_rectangle_forward = 0;
	static int*    device_object_rectangle_seen_from = 0;
	static float*  device_object_rectangle_width = 0;
	static float*  device_object_rectangle_height = 0;

	//data needed for sphere objects
	static float3* device_object_sphere_center = 0;
	static float*  device_object_sphere_radius = 0;
	static int*    device_object_sphere_seen_from = 0;

	static float3* device_path_points = 0;
	static float3* device_path_segmentTangents = 0;
	static float * device_path_segmentLengths = 0;
	
	static bool one_timer = false;

	if (!one_timer) {
		one_timer = true;
		
		cudaMalloc ((void**)(&device_object_rectangle_position), number_of_rectangle_objects*sizeof(float3));
		cudaMalloc ((void**)(&device_object_rectangle_side), number_of_rectangle_objects*sizeof(float3));
		cudaMalloc ((void**)(&device_object_rectangle_up), number_of_rectangle_objects*sizeof(float3));
		cudaMalloc ((void**)(&device_object_rectangle_forward), number_of_rectangle_objects*sizeof(float3));
		cudaMalloc ((void**)(&device_object_rectangle_seen_from), number_of_rectangle_objects*sizeof(int));
		cudaMalloc ((void**)(&device_object_rectangle_width), number_of_rectangle_objects*sizeof(float));
		cudaMalloc ((void**)(&device_object_rectangle_height), number_of_rectangle_objects*sizeof(float));

		cudaMemcpy (device_object_rectangle_position, object_rectangle_position, number_of_rectangle_objects*sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy (device_object_rectangle_side, object_rectangle_side, number_of_rectangle_objects*sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy (device_object_rectangle_up, object_rectangle_up, number_of_rectangle_objects*sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy (device_object_rectangle_forward, object_rectangle_forward, number_of_rectangle_objects*sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy (device_object_rectangle_seen_from, object_rectangle_seen_from, number_of_rectangle_objects*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy (device_object_rectangle_width, object_rectangle_width, number_of_rectangle_objects*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy (device_object_rectangle_height, object_rectangle_height, number_of_rectangle_objects*sizeof(float), cudaMemcpyHostToDevice);
	
		cudaMalloc ((void**)(&device_object_sphere_center), number_of_sphere_objects*sizeof(float3));
		cudaMalloc ((void**)(&device_object_sphere_radius), number_of_sphere_objects*sizeof(float));
		cudaMalloc ((void**)(&device_object_sphere_seen_from), number_of_sphere_objects*sizeof(int));

		cudaMemcpy (device_object_sphere_center, object_sphere_center, number_of_sphere_objects*sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy (device_object_sphere_radius, object_sphere_radius, number_of_sphere_objects*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy (device_object_sphere_seen_from, object_sphere_seen_from, number_of_sphere_objects*sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc ((void**)(&device_path_points), path_segment_count*sizeof(float3));
		cudaMalloc ((void**)(&device_path_segmentTangents), path_segment_count*sizeof(float3));
		cudaMalloc ((void**)(&device_path_segmentLengths), path_segment_count*sizeof(float));
		
		cudaMemcpy (device_path_points, path_points, path_segment_count*sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy (device_path_segmentTangents, path_segmentTangents, path_segment_count*sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy (device_path_segmentLengths, path_segmentLengths, path_segment_count*sizeof(float), cudaMemcpyHostToDevice);
	}


	
	// set up the enviroment
	dim3 block_dim (BLOCK_SIZE);
	dim3 grid_dim  (crowd_size/block_dim.x);

	// start the kernel
	determineCombinedSteering_kernel_op <<< grid_dim, block_dim >>> (
	        elapsedTime,

	        // vehicle data
	        crowd_size,
	        device_vehicle_forward,
	        device_vehicle_position,
	        device_vehicle_random,
	        device_vehicle_radius,
	        device_vehicle_side,
	        device_vehicle_up,
	        device_vehicle_speed,
	        device_vehicle_max_force,
	        device_vehicle_max_speed,
	        device_vehicle_pathDirection,

	        //data needed for rectangle objects
	        number_of_rectangle_objects,
	        device_object_rectangle_position,
	        device_object_rectangle_side,
	        device_object_rectangle_up,
	        device_object_rectangle_forward,
	        device_object_rectangle_seen_from,
	        device_object_rectangle_width,
	        device_object_rectangle_height,

	        //data needed for sphere objects
	        number_of_sphere_objects,
	        device_object_sphere_center,
	        device_object_sphere_radius,
	        device_object_sphere_seen_from,

	        // wandering
	        gWanderSwitch,
	        device_vehicle_wander_rand,

	        // path
	        gUseDirectedPathFollowing,
	        path_segment_count,
                device_path_points,
	        device_path_segmentTangents,
	        device_path_segmentLengths,
	        path_radius,
                path_length,
                path_is_cyclic,
                
	        // result
	        device_result,

	        // dirty hack
	        device_neighbor_radius,
	        device_neighbor_position,
	        device_neighbor_velocity,
	        device_neighbor_forward,
	        device_neighbor_speed
	);
	
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf( stderr, "Cuda error: KERNEL EXEC FAILED in file '%s' in line %i : %s.\n",
		__FILE__, __LINE__, cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	// read the result
	cudaMemcpy (result, device_result, crowd_size*sizeof(float3), cudaMemcpyDeviceToHost);

	// cleanup
#if 0	
	cudaFree (device_object_rectangle_position);
	cudaFree (device_object_rectangle_side);
	cudaFree (device_object_rectangle_up);
	cudaFree (device_object_rectangle_forward);
	cudaFree (device_object_rectangle_seen_from);
	cudaFree (device_object_rectangle_width);
	cudaFree (device_object_rectangle_height);
	cudaFree (device_object_sphere_center);
	cudaFree (device_object_sphere_radius);
	cudaFree (device_object_sphere_seen_from);

	cudaFree (device_path_points);
	cudaFree (device_path_segmentTangents);
	cudaFree (device_path_segmentLengths);
#endif
}

