/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef OPENSTEER_posindexpair_H
#define OPENSTEER_posindexpair_H

#include "cupp/kernel_type_binding.h"
#include "cupp/common.h"
#include "OpenSteer/deviceT/Vec3.h"


namespace OpenSteer {

/**
 * @class PosIndexPair
 * @author Jens Breitbart
 * @version 0.1
 * @date 28.08.2007
 * @brief A simple pos/index pair
 */

class PosIndexPair {
	public: /*** TYPEDEFS  ***/
		typedef PosIndexPair                  device_type;
		typedef PosIndexPair                  host_type;

	private:
		int index_;
		deviceT::Vec3 position_;

	public:

		static PosIndexPair construct (const int index, const deviceT::Vec3 &position) {
			PosIndexPair returnee;
			returnee.index_ = index;
			returnee.position_ = position;
			#if !defined(NVCC)
			//std::cout << index << std::endl;
			#endif
			return returnee;
		}

		CUPP_RUN_ON_DEVICE deviceT::Vec3 position() const {
			return position_;
		}

		CUPP_RUN_ON_DEVICE int index() const {
			return index_;
		}
};

}


#endif
