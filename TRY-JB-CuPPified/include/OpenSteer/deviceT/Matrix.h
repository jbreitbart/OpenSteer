#ifndef OPENSTEER_DEVICET_matrix_H
#define OPENSTEER_DEVICET_matrix_H

namespace OpenSteer {

class Matrix;

namespace deviceT {


struct Matrix {
	typedef OpenSteer::deviceT::Matrix        device_type;
	typedef OpenSteer::Matrix                 host_type;
	
	float elements_[ 16 ];
};


}
}

#endif
