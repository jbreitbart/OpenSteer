/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef DS_dyn_grid_clever_H
#define DS_dyn_grid_clever_H

#if defined(NVCC)
#error Not compatible with CUDA. Don't compile with nvcc.
#endif

#include <limits>
#include <functional>

// cupp
#include "cupp/kernel_type_binding.h"
#include "cupp/device_reference.h"
#include "cupp/vector.h"

#include "OpenSteer/Vec3.h"
#include "OpenSteer/PosIndexPair.h"
#include "OpenSteer/CuPPConfig.h"

#include "deviceT/dyn_grid_clever.h"

#include "dyn_grid_clever_node.h"

namespace ds {



/**
 * @class dyn_grid_clever
 * @author Jens Breitbart
 * @version 0.1
 * @date 18.09.2007
 * @platform Host only
 * @brief 
 */

class dyn_grid_clever {
	public: /*** TYPEDEFS  ***/
		typedef ds::deviceT::dyn_grid_clever                  device_type;
		typedef dyn_grid_clever                               host_type;

	private:
		/**
		 * Well ... our grid :-)
		 */
		std::vector<OpenSteer::PosIndexPair> data_;
		typedef std::vector<OpenSteer::PosIndexPair>::iterator data_iterator;
		

		std::vector<dyn_grid_clever_node> index_;
		std::vector<dyn_grid_clever_node> leaves_;

		mutable cupp::memory1d<OpenSteer::PosIndexPair> *data_d_ptr_;
		mutable cupp::memory1d<dyn_grid_clever_node> *leaves_d_ptr_;
		mutable cupp::memory1d<dyn_grid_clever_node> *area_d_ptr_;

	public:

		/**
		 * Constructor
		 * @param world_size the size of the world for each dimension <=> the world is: world_size * world_size * world_size
		 * @param number_of_cells_per_dimension how many cells you want the grid to create per dimension
		 */
		explicit dyn_grid_clever() : data_d_ptr_(0), leaves_d_ptr_(0), area_d_ptr_(0)
		{}

		~dyn_grid_clever() {
			delete data_d_ptr_;
			delete leaves_d_ptr_;
			delete area_d_ptr_;
		}

		/**
		 * This will insert an element into the grid
		 * @param inserted The element that will be inserted
		 */
		void insert ( const int insert, const int ind, const OpenSteer::deviceT::Vec3 &position ) {
			data_[insert] = OpenSteer::PosIndexPair::construct(ind, position);
			//data_.push_back( OpenSteer::PosIndexPair::construct(ind, position) );
		}
		
		void resize (const int size) {
			data_.resize(size);
		}

		void clear() {
			data_.clear();
			leaves_.clear();
		}

		void order(const int threads_per_block) {
			order(data_.begin(), data_.end(), threads_per_block,
			      -boid_worldRadius, boid_worldRadius,
			      -boid_worldRadius, boid_worldRadius,
			      -boid_worldRadius, boid_worldRadius
			     );
		}

		int size() const {
			return leaves_.size();
		}
		
	private:

		struct is_valid_left : public std::binary_function <dyn_grid_clever_node, dyn_grid_clever_node, bool> {
			bool operator() (const dyn_grid_clever_node& check, const dyn_grid_clever_node& source) const {
				return ( std::abs(source.high_x - check.low_x) >= r ) &&
				       ( std::abs(source.high_y - check.low_y) >= r ) &&
				       ( std::abs(source.high_z - check.low_z) >= r );
			}
		};

		struct is_valid_right : public std::binary_function <dyn_grid_clever_node, dyn_grid_clever_node, bool> {
			bool operator() (const dyn_grid_clever_node& check, const dyn_grid_clever_node& source) const {
				return ( std::abs(check.high_x - source.low_x) >= r ) &&
				       ( std::abs(check.high_y - source.low_y) >= r ) &&
				       ( std::abs(check.high_z - source.low_z) >= r );
			}
		};


		class vec3_average : public std::unary_function<OpenSteer::PosIndexPair, void> {
			public:
				vec3_average() : num(0) {}

				void operator() (const OpenSteer::PosIndexPair &i) {
					++num;
					sum += i.position();
				}

				OpenSteer::Vec3 result() const {
					return sum/num;
				}
			private:
				unsigned int num;
				OpenSteer::Vec3 sum;
		};
		
		template <int direction>
		struct compare_vec3 : public std::unary_function<OpenSteer::PosIndexPair, bool> {
			float position;
			
			compare_vec3 (const OpenSteer::Vec3& _position) {
				if (direction == 1) {
					position = _position.x;
				}
				if (direction == 2) {
					position = _position.y;
				}
				if (direction == 3) {
					position = _position.z;
				}
			}
			
			bool operator() (const OpenSteer::PosIndexPair& to_comp) const {
				if (direction == 1) {
					return to_comp.position().x() < position;
				}
				if (direction == 2) {
					return to_comp.position().y() < position;
				}
				if (direction == 3) {
					return to_comp.position().z() < position;
				}
			}
		};

		std::vector<float> distribution;
		
		void order(const data_iterator begin, const data_iterator end, const int threads_per_block, const float low_x, const float high_x, const float low_y, const float high_y, const float low_z, const float high_z) {

// 			std::cout << std::distance(begin, end) << " - " << threads_per_block << std::endl;
			if (std::distance(begin, end) > threads_per_block) {

				// find center of all agents between begin and end
				
				data_iterator i = begin;
				OpenSteer::Vec3 sum = OpenSteer::Vec3::zero;
				
				while (i != end) {
					OpenSteer::Vec3 pos (i->position().x(), i->position().y(), i->position().z());
					sum += pos;
					++i;
				}
				
				OpenSteer::Vec3 center = sum / std::distance(begin, end);

				const float low_x_d = std::abs(center.x - low_x);
				const float high_x_d = std::abs(high_x - center.x);
				//const float x_comp = low_x_d > high_x_d ? low_x_d : high_x_d;
				const float x_comp = low_x_d < high_x_d ? low_x_d : high_x_d;

				const float low_y_d = std::abs(center.y - low_y);
				const float high_y_d = std::abs(high_y - center.y);
				//const float y_comp = low_y_d > high_y_d ? low_y_d : high_y_d;
				const float y_comp = low_y_d < high_y_d ? low_y_d : high_y_d;

				const float low_z_d = std::abs(center.z - low_z);
				const float high_z_d = std::abs(high_z - center.z);
				//const float z_comp = low_z_d > high_z_d ? low_z_d : high_z_d;
				const float z_comp = low_z_d < high_z_d ? low_z_d : high_z_d;

				//if (div_direction ==1) {
				if (x_comp < y_comp && x_comp < z_comp) {
					//distribution.push_back(std::abs((high_x - low_x) / (high_x - center.x)));
					
					// partition the agents relative to the center
					const data_iterator x_divider = std::partition(begin, end, compare_vec3<1>(center) );

					//#pragma omp task
					order (begin, x_divider, threads_per_block, low_x, center.x, low_y, high_y, low_z, high_z);
					order (x_divider, end, threads_per_block, center.x, high_x, low_y, high_y, low_z, high_z);

					return;
				}

				//if (div_direction==2) {
				if (y_comp < x_comp && y_comp < z_comp) {
					//distribution.push_back(std::abs((high_y - low_y)));
					// partition the agents relative to the center
					const data_iterator y_divider = std::partition(begin, end, compare_vec3<2>(center) );

					//#pragma omp task
					order (begin, y_divider, threads_per_block, low_x, high_x, low_y, center.y, low_z, high_z);
					order (y_divider, end, threads_per_block, low_x, high_x, center.y, high_y, low_z, high_z);

					return;
				}

				//distribution.push_back(std::abs((high_z - low_z) / (high_z - center.z)));
				
				const data_iterator z_divider = std::partition(begin, end, compare_vec3<3>(center) );

				//#pragma omp task
				order (begin, z_divider, threads_per_block, low_x, high_x, low_y, high_y, low_z, center.z);
				order (z_divider, end, threads_per_block, low_x, high_x, low_y, high_y, center.z, high_z);
			} else {
			
				//#pragma omp critical
				leaves_.push_back ( dyn_grid_clever_node(
				                     std::distance(data_.begin(), begin),
				                     low_x, high_x, low_y, high_y, low_z, high_z,
				                     std::distance(begin, end)
				                    )
				                   );
			}
		}

	public: /*** CUPP FUNCTIONALITY  ***/

		/**
		 * @brief This function is called by the cupp::kernel_call_traits
		 * @return A on the device useable grid reference
		 */
		
		device_type transform (const cupp::device &d)  {
			if (data_d_ptr_ == 0 || data_d_ptr_ -> size() < data_.size()) {
				delete data_d_ptr_;
				data_d_ptr_ = new cupp::memory1d<OpenSteer::PosIndexPair>(d, &data_[0], data_.size());
			} else {
				data_d_ptr_ -> copy_to_device (data_.size(), &data_[0]);
			}

			if (leaves_d_ptr_ == 0 || leaves_d_ptr_ -> size() < leaves_.size()) {
				delete leaves_d_ptr_;
				leaves_d_ptr_ = new cupp::memory1d<dyn_grid_clever_node>(d, &leaves_[0], leaves_.size());
			} else {
				leaves_d_ptr_ -> copy_to_device (leaves_.size(), &leaves_[0]);
			}

			return device_type::construct(data_d_ptr_->transform(d), leaves_d_ptr_->transform(d)/*, area_d_ptr_->transform(d)*/);
		}

		/**
		 * @brief This function is called by the cupp::kernel_call_traits
		 * @return A on the device useable grid reference
		 */
		cupp::device_reference< device_type > get_device_reference(const cupp::device &d)  {
			return cupp::device_reference<device_type> (d, transform(d));
		}

		/**
		 * @brief This function is called by the kernel_call_traits
		 */
		void dirty (cupp::device_reference< device_type > /*device_copy*/) {
			throw "Should never be called";
		}

};

}

#endif
