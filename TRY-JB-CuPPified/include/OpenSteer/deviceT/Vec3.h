#ifndef OPENSTEER_DEVICET_vec3_H
#define OPENSTEER_DEVICET_vec3_H

#include "cupp/common.h"

// CUDA
#include <vector_types.h>

namespace OpenSteer {

class Vec3;

namespace deviceT {

/**
 * A dirty copy from the OpenSteer Vec3 :-D
 */
class Vec3 {
	public:
		typedef OpenSteer::deviceT::Vec3        device_type;
		typedef OpenSteer::Vec3                 host_type;
		
		float3 t;

		CUPP_RUN_ON_DEVICE float x () const {
			return t.x;
		}

		CUPP_RUN_ON_DEVICE float y () const {
			return t.y;
		}

		CUPP_RUN_ON_DEVICE float z () const {
			return t.z;
		}
		
		CUPP_RUN_ON_DEVICE Vec3 operator+ (const Vec3& v) const {
			Vec3 temp = {{t.x+v.t.x, t.y+v.t.y, t.z+v.t.z }};
			return temp;
		}
		CUPP_RUN_ON_DEVICE Vec3 operator+ (const float v) const {
			Vec3 temp = {{t.x+v, t.y+v, t.z+v }};
			return temp;
		}
		CUPP_RUN_ON_DEVICE Vec3 operator- (const Vec3& v) const {
			Vec3 temp = {{t.x-v.t.x, t.y-v.t.y, t.z-v.t.z }};
			return temp;
		}
		CUPP_RUN_ON_DEVICE Vec3 operator- (void) const          {
			Vec3 temp = {{-t.x, -t.y, -t.z}};
			return temp;
		}
		CUPP_RUN_ON_DEVICE Vec3 operator* (const float s) const {
			Vec3 temp = {{t.x*s, t.y*s, t.z*s}};
			return temp;
		}

		CUPP_RUN_ON_DEVICE Vec3 operator/ (const float s) const {
			Vec3 temp = {{t.x/s, t.y/s, t.z/s}};
			return temp;
		}

		// dot product
		CUPP_RUN_ON_DEVICE float dot (const Vec3& v) const {return (t.x * v.t.x) + (t.y * v.t.y) + (t.z * v.t.z);}

		// length squared
		CUPP_RUN_ON_DEVICE float lengthSquared () const {return this->dot (*this);}

		CUPP_RUN_ON_DEVICE float length () const {return sqrtf (lengthSquared ());}
		
		CUPP_RUN_ON_DEVICE Vec3 normalize () const {
			const float len = length ();
			return (len>0.0f) ? (*this)/len : (*this);
		}

		CUPP_RUN_ON_DEVICE bool is_zero() const { return (t.x==0.0f && t.y==0.0f && t.z==0.0f); }

		CUPP_RUN_ON_DEVICE Vec3 truncateLength (const float maxLength) const {
			const float maxLengthSquared = maxLength * maxLength;
			const float vecLengthSquared = this->lengthSquared ();
			if (vecLengthSquared <= maxLengthSquared) {
				return *this;
			} else {
				return (*this) * (maxLength / sqrtf (vecLengthSquared));
			}
		}
		
		CUPP_RUN_ON_DEVICE void cross(const Vec3& a, const Vec3& b) {
			t.x = (a.t.y * b.t.z) - (a.t.z * b.t.y);
			t.y = (a.t.z * b.t.x) - (a.t.x * b.t.z);
			t.z = (a.t.x * b.t.y) - (a.t.y * b.t.x);
		}
		
		CUPP_RUN_ON_DEVICE Vec3 sphericalWrapAround (float radius) {
			const float r = this -> length();
			if (r > radius) {
				return *this + ((*this/r) * radius * -2);
			} else {
				return *this;
			}
		}
		
		CUPP_RUN_ON_DEVICE Vec3 parallelComponent (const Vec3& unitBasis) const {
			const float projection = this->dot (unitBasis);
			return unitBasis * projection;
		}

		CUPP_RUN_ON_DEVICE Vec3 perpendicularComponent (const Vec3& unitBasis) const {
			return (*this) - parallelComponent (unitBasis);
		}
};


}
}

#endif
