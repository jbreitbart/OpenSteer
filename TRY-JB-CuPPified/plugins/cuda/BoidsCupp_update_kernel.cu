#include "cupp/deviceT/vector.h"
#include "cupp/common.h"

#include "OpenSteer/deviceT/Vec3.h"
#include "OpenSteer/deviceT/Boid.h"
#include "OpenSteer/deviceT/Matrix.h"
#include "OpenSteer/CuPPConfig.h"
#include "OpenSteer/kernels.h"

using OpenSteer::deviceT::Vec3;
using OpenSteer::deviceT::Boid;
using OpenSteer::deviceT::Matrix;


template<class T>
__device__ T interpolate (const float alpha, const T& x0, const T& x1) {
	return x0 + ((x1 - x0) * alpha);
}

__device__ float clip (const float x, const float min, const float max) {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

template<class T>
__device__ void blendIntoAccumulator (const float smoothRate, const T& newValue, T& smoothedAccumulator)
{
	smoothedAccumulator = interpolate (clip (smoothRate, 0, 1), smoothedAccumulator, newValue);
}

__device__ Vec3 limitMaxDeviationAngle (const Vec3& source, const float cosineOfConeAngle, const Vec3& basis) {
	// immediately return zero length input vectors
	float sourceLength = source.length();
	if (sourceLength == 0) return source;

	// measure the angular diviation of "source" from "basis"
	const Vec3 direction = source / sourceLength;
	float cosineOfSourceAngle = direction.dot (basis);

	// Simply return "source" if it already meets the angle criteria.
	// (note: we hope this top "if" gets compiled out since the flag
	// is a constant when the function is inlined into its caller)
	// source vector is already inside the cone, just return it
	if (cosineOfSourceAngle >= cosineOfConeAngle) return source;

	// find the portion of "source" that is perpendicular to "basis"
	const Vec3 perp = source.perpendicularComponent (basis);

	// normalize that perpendicular
	const Vec3 unitPerp = perp.normalize ();

	// construct a new vector whose length equals the source vector,
	// and lies on the intersection of a plane (formed the source and
	// basis vectors) and a cone (whose axis is "basis" and whose
	// angle corresponds to cosineOfConeAngle)
	float perpDist = sqrtf (1 - (cosineOfConeAngle * cosineOfConeAngle));
	const Vec3 c0 = basis * cosineOfConeAngle;
	const Vec3 c1 = unitPerp * perpDist;
	return (c0 + c1) * sourceLength;
}



__global__ void update (const float                           elapsedTime,
                              cupp::deviceT::vector <Vec3>   &positions_,
                              cupp::deviceT::vector <Vec3>   &forwards_,
                        const cupp::deviceT::vector <Vec3>   &steering_forces_,
                              cupp::deviceT::vector <Boid>   &boids_,
                              cupp::deviceT::vector <Matrix> &render_position_
                       )
{
	const unsigned int my_index        = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ Vec3 steering_forces[threads_per_block];
	steering_forces[threadIdx.x] = steering_forces_[my_index];
	Vec3 &force = steering_forces[threadIdx.x];

	__shared__ Boid boids[threads_per_block];
	boids[threadIdx.x] = boids_[my_index];
	Boid &me    = boids[threadIdx.x];

	//__shared__ Vec3 positions[threads_per_block];
	//positions[threadIdx.x] = positions_[my_index];
	Vec3 &position    = positions_[my_index];//positions[threadIdx.x];

	//__shared__ Vec3 forwards[threads_per_block];
	//forwards[threadIdx.x] = forwards_[my_index];
	Vec3 &forward    = forwards_[my_index];//forwards[threadIdx.x];

	// adjustRawSteeringForce
	if (!((me.speed > boid_maxAdjustedSpeed) || force.is_zero())) {
		const float range = me.speed / boid_maxAdjustedSpeed;
		const float cosine = interpolate (powf (range, 20), 1.0f, -1.0f);
		force = limitMaxDeviationAngle (force, cosine, forward);
	}

	// enforce limit on magnitude of steering force
	force = force.truncateLength (boid_maxForce);

	// compute acceleration and velocity
	const Vec3 newAcceleration = force; /* / mass; mass == 1.0f */
	Vec3 newVelocity = forward * me.speed;

	// damp out abrupt changes and oscillations in steering acceleration
	// (rate is proportional to time step, then clipped into useful range)
	if (elapsedTime > 0) {
		const float smoothRate = clip (9.0f * elapsedTime, 0.15f, 0.4f);
		blendIntoAccumulator (smoothRate, newAcceleration, me.smoothedAcceleration);
		
		// Euler integrate (per frame) acceleration into velocity
		newVelocity = newVelocity + me.smoothedAcceleration * elapsedTime;
	}

	// enforce speed limit
	newVelocity = newVelocity.truncateLength (boid_maxSpeed);

	// update Speed
	me.speed =  newVelocity.length();

	const Vec3 globalUp = { 0.0f, 0.2f, 0.0f};

	const Vec3 accelUp = me.smoothedAcceleration * 0.05f;

	const Vec3 bankUp = accelUp + globalUp;

	const float smoothRate = elapsedTime * 3;
	Vec3 tempUp = me.up;
	blendIntoAccumulator (smoothRate, bankUp, tempUp);
	me.up = tempUp.normalize();

	if (me.speed > 0.0f) {
		const Vec3 newUnitForward = newVelocity / me.speed;
		forward = newUnitForward;
	}

	// derive new side basis vector from NEW forward and OLD up
	me.side.cross (forward, me.up);

	me.up.cross (me.side, forward);

	// Euler integrate (per frame) velocity into position

	position = position + (newVelocity * elapsedTime);
	position = position.sphericalWrapAround (boid_worldRadius);
	
	boids_[my_index] = me;
	positions_[my_index] = position;
	forwards_[my_index] = forward;
	render_position_[my_index].elements_[0] = me.side.t.x;
	render_position_[my_index].elements_[1] = me.side.t.y;
	render_position_[my_index].elements_[2] = me.side.t.z;
	//render_position_[my_index].elements_[3] = 0.0f;
	render_position_[my_index].elements_[4] = me.up.t.x;
	render_position_[my_index].elements_[5] = me.up.t.y;
	render_position_[my_index].elements_[6] = me.up.t.z;
	//render_position_[my_index].elements_[7] = 0.0f;
	render_position_[my_index].elements_[8] = forward.t.x;
	render_position_[my_index].elements_[9] = forward.t.y;
	render_position_[my_index].elements_[10] = forward.t.z;
	//render_position_[my_index].elements_[11] = 0.0f;
	render_position_[my_index].elements_[12] = position.t.x;
	render_position_[my_index].elements_[13] = position.t.y;
	render_position_[my_index].elements_[14] = position.t.z;
	//render_position_[my_index].elements_[15] = 1.0f;
}



#if 0
slow ... no idea why
__global__ void update (const float                           elapsedTime,
                              cupp::deviceT::vector <Vec3>   &positions_,
                              cupp::deviceT::vector <Vec3>   &forwards_,
                        const cupp::deviceT::vector <Vec3>   &steering_forces_,
                              cupp::deviceT::vector <Boid>   &boids_,
                              cupp::deviceT::vector <Matrix> &render_position_
                       )
{
	const unsigned int my_index        = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ Vec3 steering_forces[threads_per_block];
	steering_forces[threadIdx.x] = steering_forces_[my_index];
	Vec3 &force = steering_forces[threadIdx.x];

	__shared__ Boid boids[threads_per_block];
	boids[threadIdx.x] = boids_[my_index];
	Boid &me    = boids[threadIdx.x];

	Vec3 &position    = positions_[my_index];

	//__shared__ Vec3 forwards[threads_per_block];
	//forwards[threadIdx.x] = forwards_[my_index];
	Vec3 &forward    = forwards_[my_index];//forwards[threadIdx.x];


	// adjustRawSteeringForce
	if (!((me.speed > maxAdjustedSpeed) || force.is_zero())) {
		const float range = me.speed / maxAdjustedSpeed;
		const float cosine = interpolate (powf (range, 20), 1.0f, -1.0f);
		force = limitMaxDeviationAngle (force, cosine, forward);
	}

	// enforce limit on magnitude of steering force
	force = force.truncateLength (maxForce);

	// compute acceleration and velocity
	const Vec3 newAcceleration = force; /* / mass; mass == 1.0f */
	Vec3 newVelocity = forward * me.speed;

	// damp out abrupt changes and oscillations in steering acceleration
	// (rate is proportional to time step, then clipped into useful range)
	if (elapsedTime > 0) {
		const float smoothRate = clip (9.0f * elapsedTime, 0.15f, 0.4f);
		blendIntoAccumulator (smoothRate, newAcceleration, me.smoothedAcceleration);
		
		// Euler integrate (per frame) acceleration into velocity
		newVelocity = newVelocity + me.smoothedAcceleration * elapsedTime;
	}

	// enforce speed limit
	newVelocity = newVelocity.truncateLength (maxSpeed);

	// update Speed
	me.speed =  newVelocity.length();

	const Vec3 globalUp = { 0.0f, 0.2f, 0.0f};

	const Vec3 accelUp = me.smoothedAcceleration * 0.05f;

	const Vec3 bankUp = accelUp + globalUp;

	const float smoothRate = elapsedTime * 3;
	
	Vec3 tempUp = {render_position_[my_index].elements_[4] ,render_position_[my_index].elements_[5], render_position_[my_index].elements_[6]};
	
	blendIntoAccumulator (smoothRate, bankUp, tempUp);
	tempUp = tempUp.normalize();

	if (me.speed > 0.0f) {
		const Vec3 newUnitForward = newVelocity / me.speed;
		forward = newUnitForward;
	}

	// derive new side basis vector from NEW forward and OLD up
	Vec3 side;
	side.cross (forward, tempUp);

	tempUp.cross (side, forward);

	// Euler integrate (per frame) velocity into position

	position = position + (newVelocity * elapsedTime);
	position = position.sphericalWrapAround (worldRadius);
	
	boids_[my_index] = me;
	positions_[my_index] = position;
	forwards_[my_index] = forward;
	render_position_[my_index].elements_[0] = side.t.x;
	render_position_[my_index].elements_[1] = side.t.y;
	render_position_[my_index].elements_[2] = side.t.z;
	//render_position_[my_index].elements_[3] = 0.0f;
	render_position_[my_index].elements_[4] = tempUp.t.x;
	render_position_[my_index].elements_[5] = tempUp.t.y;
	render_position_[my_index].elements_[6] = tempUp.t.z;
	//render_position_[my_index].elements_[7] = 0.0f;
	render_position_[my_index].elements_[8] = forward.t.x;
	render_position_[my_index].elements_[9] = forward.t.y;
	render_position_[my_index].elements_[10] = forward.t.z;
	//render_position_[my_index].elements_[11] = 0.0f;
	render_position_[my_index].elements_[12] = position.t.x;
	render_position_[my_index].elements_[13] = position.t.y;
	render_position_[my_index].elements_[14] = position.t.z;
	//render_position_[my_index].elements_[15] = 1.0f;
}
#endif


update_kernelT get_update_kernel() {
	return (update_kernelT)update;
}
