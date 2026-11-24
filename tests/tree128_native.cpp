#include "../data_loader/cpp/tree128_generated.h"
#include <cstdint>
struct Raw {
    const std::int16_t* p;
    int get(int i) const { return p[i]; }
};
extern "C" void tree128_select_many(const std::int16_t* raw, int size, std::int16_t* out) {
    for (int i=0; i<size; ++i) {
        Raw c{raw+8*i};
        out[i] = frozen_tree128::select(c);
    }
}
