/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef DS_DEVICET_grid_H
#define DS_DEVICET_grid_H

#include "cupp/common.h"

#include "OpenSteer/deviceT/Vec3.h"

#include <vector_types.h>

namespace ds {

template <typename T>
class grid;

namespace deviceT {

/**
 * @class grid
 * @author Jens Breitbart
 * @version 0.1
 * @date 08.08.2007
 * @platform Host only
 * @brief A very simple 3-D grid implementation.
 * Requirements for template parameter T:
 * - T.position().{x|y|z} needs to give the {x|y|z}-coord of the position of the object
 */

using OpenSteer::deviceT::Vec3;

template <typename T>
class grid {
	public: /*** TYPEDEFS  ***/
		typedef grid<T>         device_type;
		typedef ds::grid<T>     host_type;

	public:
		/**
		 * Our grid in a 1D memory
		 */
		cupp::deviceT::memory1d<T> data_;

		/**
		 * The index for every block field, used to access
		 */
		cupp::deviceT::memory1d<int> index_;
		
		/**
		 * Dimension of a cell in each dimension
		 */
		float cell_size_;

		/**
		 * The number of cells in each dimension; total number of cells == number_of_cells_per_dimension_³
		 */
		int number_of_cells_per_dimension_;

		float world_size_;

	public:
		CUPP_RUN_ON_DEVICE int get_index (const int3& position) const {
			int where_to_insert  = position.x;
			where_to_insert     += position.y * number_of_cells_per_dimension_;
			where_to_insert     += position.z * number_of_cells_per_dimension_ * number_of_cells_per_dimension_;

			return where_to_insert;
		}

		/**
		 * Constructor
		 * @param world_size the size of the world for each dimension <=> the world is: world_size * world_size * world_size
		 * @param number_of_cells_per_dimension how many cells you want the grid to create per dimension
		 */
		CUPP_RUN_ON_HOST static grid<T> construct(const cupp::deviceT::memory1d<T> &data, cupp::deviceT::memory1d<int> index, const float cell_size, const int number_of_cells_per_dimension, const float world_size) {
			grid returnee;
			returnee.data_ = data;
			returnee.index_ = index;
			returnee.cell_size_ = cell_size;
			returnee.number_of_cells_per_dimension_ = number_of_cells_per_dimension;
			returnee.world_size_ = world_size;

			return returnee;
			
		}

		CUPP_RUN_ON_DEVICE int3 get_cell_index (const Vec3& position) const {
			return make_int3 ( static_cast <int> ( (position.x()+world_size_) / cell_size_),
			                   static_cast <int> ( (position.y()+world_size_) / cell_size_),
			                   static_cast <int> ( (position.z()+world_size_) / cell_size_)
			                 );
		}

		/**
		 * Returns a pointer to the beginning of the requested cell
		 */
		CUPP_RUN_ON_DEVICE const T* begin(const int3& cell_index) const {
			return &data_[ index_[get_index(cell_index)] ];
		}
		CUPP_RUN_ON_DEVICE const T* begin(const int3& cell_index, const int indexes[]) const {
			return &data_[ indexes[get_index(cell_index)] ];
		}

		CUPP_RUN_ON_DEVICE const T* begin(const int3& cell_index, const int indexes[], const int data[]) const {
			return &data[ indexes[get_index(cell_index)] ];
		}

		/**
		 * Returns a pointer to end + 1 of the requested cell
		 */
		CUPP_RUN_ON_DEVICE const T* end(const int3& cell_index) const {
			return &data_[ index_[get_index(cell_index)+1]];
		}
		CUPP_RUN_ON_DEVICE const T* end(const int3& cell_index, const int indexes[]) const {
			return &data_[ indexes[get_index(cell_index)+1]];
		}
		CUPP_RUN_ON_DEVICE const T* end(const int3& cell_index, const int indexes[], const int data[]) const {
			return &data[ indexes[get_index(cell_index)+1]];
		}
		

		CUPP_RUN_ON_DEVICE float cell_size() const {
			return cell_size_;
		}

		CUPP_RUN_ON_DEVICE int number_of_cells_per_dimension() const {
			return number_of_cells_per_dimension_;
		}

		CUPP_RUN_ON_DEVICE int get_index(const int i) {
			return index_[i];
		}
};

}
}


#endif
