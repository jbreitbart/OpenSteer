#ifndef OPENSTEER_boid_H
#define OPENSTEER_boid_H

#include "OpenSteer/CuPPConfig.h"
#include "OpenSteer/deviceT/Boid.h"

#include "cupp/device.h"

#include <vector>

namespace OpenSteer {

class Boid {
	public:
		typedef deviceT::Boid device_type;
		typedef Boid          host_type;

		device_type transform(const cupp::device &d) {
			device_type returnee;
			returnee.up = up_.transform(d);
			returnee.side = side_.transform(d);
			returnee.speed = speed_;
			returnee.smoothedAcceleration = smoothedAcceleration_.transform(d);
			return returnee;
		}

		Boid& operator= (const device_type &rhs) {
			up_ = rhs.up;
			side_ = rhs.side;
			speed_ = rhs.speed;
			smoothedAcceleration_ = rhs.smoothedAcceleration;
			return *this;
		}
		
		// type for a flock: an STL vector of Boid pointers
		typedef cupp::vector<Boid>       groupType;
		typedef std::vector<Boid*>       AVGroup;
		typedef AVGroup::const_iterator  AVIterator;
		

		/**
		 * Default constructor
		 * @param pcontainer The proximity datastructure this boid lives in
		 */
		Boid(kapaga::random_number_source const& _rand_source, const Vec3 &forward) {
			reset(_rand_source, forward);
		}

		/**
		 * Copy Constructor
		 */
		Boid(const Boid &c) :
		up_(c.up_), side_(c.side_), speed_(c.speed_)
		{ }

		void reset (kapaga::random_number_source const &_rand_source, const Vec3 &forward) {
			speed_ = boid_maxSpeed * 0.3f;

			up_.set (0, 1, 0);

			smoothedAcceleration_ = Vec3::zero;

			// Only @c reset needs to draw random numbers in some function calls.
			kapaga::random_number_source rand_source( _rand_source );

			// bknafla: adapt to thread safe random number generation.
			// randomize initial orientation
			regenerateOrthonormalBasisUF( forward );

		}

	private:
		
		/**
		 * Doing math stuff
		 */
		void regenerateOrthonormalBasisUF (const Vec3& forward) {
			// derive new side basis vector from NEW forward and OLD up
			side_.cross (forward, up_);

			up_.cross (side_, forward);
		}

	private:
		Vec3 up_;
		Vec3 side_;
		Vec3 smoothedAcceleration_;
		
		float speed_;
};

}

#endif
