#ifndef OPENSTEER_DEVICET_boid_H
#define OPENSTEER_DEVICET_boid_H

#include "OpenSteer/deviceT/Vec3.h"

namespace OpenSteer {

class Boid;

namespace deviceT {


struct Boid {
	typedef OpenSteer::deviceT::Boid        device_type;
	typedef OpenSteer::Boid                 host_type;
	
	Vec3 up;
	Vec3 side;
	Vec3 smoothedAcceleration;
	
	float speed;
};


}
}

#endif
