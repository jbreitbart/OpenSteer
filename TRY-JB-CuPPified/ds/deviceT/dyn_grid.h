/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef DS_DEVICET_dyn_grid_H
#define DS_DEVICET_dyn_grid_H

#include "cupp/common.h"

#include "OpenSteer/deviceT/Vec3.h"
#include "OpenSteer/PosIndexPair.h"

#include "ds/dyn_grid_node.h"

#include <vector_types.h>

namespace ds {

class dyn_grid;

namespace deviceT {

/**
 * @class dyn_grid
 * @author Jens Breitbart
 * @version 0.1
 * @date 16.09.2007
 */

using OpenSteer::deviceT::Vec3;

class dyn_grid {
	public: /*** TYPEDEFS  ***/
		typedef dyn_grid         device_type;
		typedef ds::dyn_grid     host_type;

		
		cupp::deviceT::memory1d<OpenSteer::PosIndexPair> data_;
		cupp::deviceT::memory1d<dyn_grid_node> leaves_;
		cupp::deviceT::memory1d<dyn_grid_node> area_;

		CUPP_RUN_ON_DEVICE OpenSteer::PosIndexPair get_PosIndex (const dyn_grid_node& n, const int index) const {
			return data_[n.index + index];
		}

		CUPP_RUN_ON_DEVICE dyn_grid_node get_block_data(const int index) const {
			return leaves_[index];
		}

		CUPP_RUN_ON_DEVICE dyn_grid_node get_area(const int index) const {
			return area_[index];
		}

		CUPP_RUN_ON_DEVICE unsigned int get_index(const dyn_grid_node &n, const int index) const {
			return data_[n.index + index].index();
		}

		CUPP_RUN_ON_DEVICE Vec3 get_position(const dyn_grid_node &n, const int index) const {
			
			return data_[n.index + index].position();
		}
		
		/**
		 * Constructor
		 * @param world_size the size of the world for each dimension <=> the world is: world_size * world_size * world_size
		 * @param number_of_cells_per_dimension how many cells you want the grid to create per dimension
		 */
		CUPP_RUN_ON_HOST static dyn_grid construct(const cupp::deviceT::memory1d<OpenSteer::PosIndexPair> &data, cupp::deviceT::memory1d<dyn_grid_node> leaves, const cupp::deviceT::memory1d<dyn_grid_node> &area) {
			dyn_grid returnee;
			returnee.data_ = data;
			returnee.leaves_ = leaves;
			returnee.area_ = area;

			return returnee;
			
		}
};

}
}


#endif
