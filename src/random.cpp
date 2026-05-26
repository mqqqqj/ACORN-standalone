#include "acorn/random.h"

namespace acorn {

RandomGenerator::RandomGenerator(int64_t seed) : mt((unsigned int)seed) {}

int RandomGenerator::rand_int() {
    return mt() & 0x7fffffff;
}

int64_t RandomGenerator::rand_int64() {
    return int64_t(rand_int()) | int64_t(rand_int()) << 31;
}

int RandomGenerator::rand_int(int max) {
    return mt() % max;
}

float RandomGenerator::rand_float() {
    return mt() / float(mt.max());
}

double RandomGenerator::rand_double() {
    return mt() / double(mt.max());
}

} // namespace acorn
