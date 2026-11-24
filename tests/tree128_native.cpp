#include "../data_loader/cpp/tree128_generated.h"
#include <cstdint>
struct Raw {
    const std::int16_t* p;
    int get(int i) const { return p[i]; }
};
extern "C" void tree128_select_many(const std::int16_t* raw, int size, std::int16_t* out) {
    for (int i=0; i<size; ++i) {
        Raw c{raw+8*i};
        out[i*4] = frozen_tree128::select_reference(c);
        out[i*4+1] = frozen_tree128::select_binary(c);
        out[i*4+2] = frozen_tree128::select_small_switch(c);
        out[i*4+3] = frozen_tree128::select_mixed(c);
    }
}
extern "C" int tree128_check_cartesian() {
    constexpr int sizes[8] = {16,4,6,16,4,6,8,5};
    std::int16_t raw[8] = {};
    do {
        std::int16_t out[4];
        tree128_select_many(raw, 1, out);
        if (out[0]<0 || out[0]>=128 || out[1]!=out[0] || out[2]!=out[0] || out[3]!=out[0]) return 0;
        int j=7;
        for (; j>=0; --j) { if (++raw[j]<sizes[j]) break; raw[j]=0; }
        if (j<0) break;
    } while (true);
    return 1;
}
