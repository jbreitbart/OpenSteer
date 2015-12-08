#ifndef OPENSTEER_kernel_H
#define OPENSTEER_kernel_H

#include "cupp/common.h"
#include "cupp/deviceT/vector.h"

#include "OpenSteer/deviceT/Vec3.h"
#include "OpenSteer/deviceT/Matrix.h"
#include "OpenSteer/deviceT/Boid.h"
#include "OpenSteer/PosIndexPair.h"

#include "ds/deviceT/grid.h"
#include "ds/deviceT/gpu_grid.h"
#include "ds/deviceT/dyn_grid.h"
#include "ds/deviceT/dyn_grid_clever.h"
#include "ds/deviceT/dyn_grid_clever_v2.h"

/*** find neighbours kernel ***/
typedef void(*
              find_neighbours_kernelT
            ) (const cupp::deviceT::vector< OpenSteer::deviceT::Vec3>&,
               const float,
               cupp::deviceT::vector<int>                            &
              );

find_neighbours_kernelT get_find_neighbours_kernel();

/*** simulate kernel ***/
typedef void(*
              simulate_kernelT
            ) (const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >&,
               const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >&,
                     cupp::deviceT::vector< OpenSteer::deviceT::Vec3 > &
              );


simulate_kernelT get_find_neighbours_simulate_kernel();


/*** simulate + think frequency kernel ***/
typedef void (*
               simulate_frequency_kernelT
             ) (const int,
                const unsigned int,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3 >&,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3 >&,
                cupp::deviceT::vector< OpenSteer::deviceT::Vec3 >      &
               );


simulate_frequency_kernelT get_find_neighbours_simulate_frequency_kernel();


/*** simulate + think frequency + grid kernel ***/
typedef void (*
               simulate_frequency_grid_kernelT
             ) (const int,
                const ds::deviceT::grid<int> &,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >  &,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >  &,
                      cupp::deviceT::vector< OpenSteer::deviceT::Vec3 >   &
               );

simulate_frequency_grid_kernelT get_find_neighbours_simulate_frequency_grid_kernel();


/*** simulate + think frequency + gpu grid kernel ***/
typedef void (*
               simulate_frequency_gpu_grid_kernelT
             ) (const int,
                const ds::deviceT::gpu_grid &,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >  &,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >  &,
                      cupp::deviceT::vector< OpenSteer::deviceT::Vec3 >   &
               );

simulate_frequency_gpu_grid_kernelT get_find_neighbours_simulate_frequency_gpu_grid_kernel();


/*** simulate + dyn grid kernel ***/
typedef void (*
               find_neighbours_simulate_dyn_gr_kernelT
             ) (const ds::deviceT::dyn_grid                               &,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >  &,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >  &,
                      cupp::deviceT::vector< OpenSteer::deviceT::Vec3 >   &
               );

find_neighbours_simulate_dyn_gr_kernelT get_find_neighbours_simulate_dyn_grid_kernel();


/*** simulate + dyn grid clever kernel ***/
typedef void (*
              find_neighbours_simulate_dyn_gr__clever_kernelT
             ) (const ds::deviceT::dyn_grid_clever    &,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >  &,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >  &,
                      cupp::deviceT::vector< OpenSteer::deviceT::Vec3 >   &
               );

find_neighbours_simulate_dyn_gr__clever_kernelT get_find_neighbours_simulate_dyn_grid_clever_kernel();


/*** simulate + dyn grid clever kernel ***/
typedef void (*
              find_neighbours_simulate_dyn_gr__clever_tf_kernelT
             ) (const ds::deviceT::dyn_grid_clever    &,
                const ds::deviceT::dyn_grid_clever    &,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >  &,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >  &,
                      cupp::deviceT::vector< OpenSteer::deviceT::Vec3 >   &
               );

find_neighbours_simulate_dyn_gr__clever_tf_kernelT get_find_neighbours_simulate_dyn_grid_clever_tf_kernel();

/*** simulate + dyn grid clever V2 kernel ***/
typedef void (*
               find_neighbours_simulate_dyn_gr_clever_v2_kernelT
             ) (const ds::deviceT::dyn_grid_clever_v2                     &,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >  &,
                const cupp::deviceT::vector< OpenSteer::deviceT::Vec3  >  &,
                      cupp::deviceT::vector< OpenSteer::deviceT::Vec3 >   &
               );

find_neighbours_simulate_dyn_gr_clever_v2_kernelT get_find_neighbours_simulate_dyn_grid_clever_v2_kernel();


/***  preprocess grid kernel  ***/
typedef void (*
               preprocess_gridT
             ) ( const cupp::deviceT::vector< OpenSteer::deviceT::Vec3 >  &,
                       ds::deviceT::gpu_grid                              &,
                       unsigned int
               );

preprocess_gridT get_preprocess_grid_kernel();


/***  count V1  ***/
typedef void (*
               v1_countT
             ) ( const cupp::deviceT::vector< OpenSteer::deviceT::Vec3 >  &,
                       ds::deviceT::gpu_grid                              &,
                       unsigned int
               );

v1_countT get_v1_count_kernel();


/***  prescan V1  ***/
typedef void  (*
                v1_prescanT
              ) ( ds::deviceT::gpu_grid        &,
                  unsigned int
                );

v1_prescanT get_v1_prescan_kernel();


/***  fill V1  ***/
typedef void (*
               v1_fillT
             ) ( const cupp::deviceT::vector< OpenSteer::deviceT::Vec3 >  &,
                       ds::deviceT::gpu_grid                              &,
                       unsigned int
               );

v1_fillT get_v1_fill_kernel();


/***  update kernel  ***/
typedef void (*
               update_kernelT
             ) (const float,
                      cupp::deviceT::vector <OpenSteer::deviceT::Vec3>&,
                      cupp::deviceT::vector <OpenSteer::deviceT::Vec3>&,
                const cupp::deviceT::vector <OpenSteer::deviceT::Vec3>&,
                      cupp::deviceT::vector <OpenSteer::deviceT::Boid>&,
                      cupp::deviceT::vector <OpenSteer::deviceT::Matrix>&
               );

update_kernelT get_update_kernel();


#endif
