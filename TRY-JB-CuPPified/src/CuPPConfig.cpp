#include "cupp/device.h"
#include "OpenSteer/CuPPConfig.h"

namespace OpenSteer {

const cupp::device cupp_device;

const unsigned int no_of_multiprocessors = 12;
const unsigned int think_frequency = 10;

const size_t gBoidStartCount = 2048*10*2;//8192*4*4;//no_of_multiprocessors * threads_per_block * think_frequency;
const size_t add_vehicle_count = 1024;//threads_per_block * think_frequency;

}
