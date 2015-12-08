/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef DS_DEVICET_dyn_grid_clever_v2_H
#define DS_DEVICET_dyn_grid_clever_v2_H

#include "cupp/common.h"

#include "OpenSteer/deviceT/Vec3.h"
#include "OpenSteer/PosIndexPair.h"

#include "ds/dyn_grid_clever_v2_node.h"

#include <vector_types.h>

namespace ds {

class dyn_grid_clever_v2;

namespace deviceT {

/**
 * @class dyn_grid_clever_v2
 * @author Jens Breitbart
 * @version 0.1
 * @date 01.10.2007
 */

using OpenSteer::deviceT::Vec3;

class dyn_grid_clever_v2 {
	public: /*** TYPEDEFS  ***/
		typedef dyn_grid_clever_v2         device_type;
		typedef ds::dyn_grid_clever_v2     host_type;

		
		cupp::deviceT::memory1d<OpenSteer::PosIndexPair> data_;
		cupp::deviceT::memory1d<dyn_grid_clever_v2_node> leaves_;
		cupp::deviceT::memory1d<dyn_grid_clever_v2_node> area_;

		CUPP_RUN_ON_DEVICE OpenSteer::PosIndexPair get_PosIndex (const dyn_grid_clever_v2_node& n, const int index) const {
			return data_[n.index + index];
		}

		CUPP_RUN_ON_DEVICE dyn_grid_clever_v2_node get_block_data(const int index) const {
			return leaves_[index];
		}
		

		CUPP_RUN_ON_DEVICE dyn_grid_clever_v2_node get_area(const int index) const {
			return area_[index];
		}

		CUPP_RUN_ON_DEVICE unsigned int get_index(const dyn_grid_clever_v2_node &n, const int index) const {
			return data_[n.index + index].index();
		}

		CUPP_RUN_ON_DEVICE Vec3 get_position(const dyn_grid_clever_v2_node &n, const int index) const {
			
			return data_[n.index + index].position();
		}
		
		/**
		 * Constructor
		 * @param world_size the size of the world for each dimension <=> the world is: world_size * world_size * world_size
		 * @param number_of_cells_per_dimension how many cells you want the grid to create per dimension
		 */
		CUPP_RUN_ON_HOST static dyn_grid_clever_v2 construct(const cupp::deviceT::memory1d<OpenSteer::PosIndexPair> &data, cupp::deviceT::memory1d<dyn_grid_clever_v2_node> leaves, const cupp::deviceT::memory1d<dyn_grid_clever_v2_node> &area) {
			dyn_grid_clever_v2 returnee;
			returnee.data_ = data;
			returnee.leaves_ = leaves;
			returnee.area_ = area;

			return returnee;
			
		}
};

}
}


#endif
