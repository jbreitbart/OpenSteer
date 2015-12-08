/// @note header stuff missing

#include <vector_types.h>     // CUDA vector types
#include <vector_functions.h> // CUDA vector types

inline __device__ __host__ float3 operator- (const float3 lhs) {
	return make_float3 (-lhs.x, -lhs.y, -lhs.z);
}

inline __device__ __host__ float3 operator+ (const float3 lhs, const float3 rhs) {
	return make_float3 (lhs.x+rhs.x, lhs.y+rhs.y, lhs.z+rhs.z);
}
inline __device__ __host__ float3 operator- (const float3 lhs, const float3 rhs) {
	return make_float3 (lhs.x-rhs.x, lhs.y-rhs.y, lhs.z-rhs.z);
}
inline __device__ __host__ float3 operator* (const float3 lhs, const float3 rhs) {
	return make_float3 (lhs.x*rhs.x, lhs.y*rhs.y, lhs.z*rhs.z);
}
inline __device__ __host__ float3 operator/ (const float3 lhs, const float3 rhs) {
	return make_float3 (lhs.x/rhs.x, lhs.y/rhs.y, lhs.z/rhs.z);
}

inline __device__ __host__ bool operator== (const float3 lhs, const float3 rhs) {
	return lhs.x==rhs.x && lhs.y==rhs.y && lhs.z==rhs.z;
}

inline __device__ __host__ bool operator!= (const float3 lhs, const float3 rhs) {
	return !(lhs == rhs);
}

inline __device__ __host__ float3 operator+ (const float3 lhs, const float rhs) {
	return make_float3 (lhs.x+rhs, lhs.y+rhs, lhs.z+rhs);
}
inline __device__ __host__ float3 operator- (const float3 lhs, const float rhs) {
	return make_float3 (lhs.x-rhs, lhs.y-rhs, lhs.z-rhs);
}
inline __device__ __host__ float3 operator* (const float3 lhs, const float rhs) {
	return make_float3 (lhs.x*rhs, lhs.y*rhs, lhs.z*rhs);
}
inline __device__ __host__ float3 operator/ (const float3 lhs, const float rhs) {
	return make_float3 (lhs.x/rhs, lhs.y/rhs, lhs.z/rhs);
}

inline __device__ __host__ float dot (float3 a, float3 b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __device__ __host__ float length_squared (const float3 temp) {
	return dot (temp, temp);
}

inline __device__ __host__ float length (const float3 temp) {
	return sqrtf( length_squared(temp) );
}

inline __device__ __host__ float3 normalize (const float3 temp) {
	// skip divide if length is zero
	const float len = length (temp);
	return (len>0.0f) ? temp/len : temp;
}

// return component of vector parallel to a unit basis vector
// (IMPORTANT NOTE: assumes "basis" has unit magnitude (length==1))
inline __device__ __host__ float3 parallel_component (const float3 vec, const float3 unit_basis) {
	const float projection = dot(vec, unit_basis);
	return unit_basis * projection;
}

// return component of vector perpendicular to a unit basis vector
// (IMPORTANT NOTE: assumes "basis" has unit magnitude (length==1))
inline __device__ __host__ float3 perpendicular_component (const float3 vec, const float3 unit_basis)
{
	return vec - parallel_component (vec, unit_basis);
}

/**
 *  Retuns distance between @a a and @a b.
 */
inline __device__ __host__ float distance (const float3 a, const float3 b)
{
	return length(a-b);
}

