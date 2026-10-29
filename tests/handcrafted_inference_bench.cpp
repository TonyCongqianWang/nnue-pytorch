// Inference proxy: matched synthetic weights/inputs, full route dispatch, no training.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <vector>
#include "attacks.h"
#include "nnue/nnue_architecture.h"
#include "position.h"
using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;

// Straightforward separate-bank reference, identical activation and score arithmetic.
static i32 separate(const NetworkArchitecture& a,
                    const NetworkArchitecture& b,
                    const u8*                  input,
                    const NNZInfo<L1>&         nnz) {
    alignas(64) i32 first[2][16], second[2][16], joined[32], out[2][16];
    alignas(64) u8  h[128];
    a.fc_0.propagate(input, first[0], nnz);
    b.fc_0.propagate(input, first[1], nnz);
    const i64 skips = i64(first[0][14]) - a.fc_0.get_biases()[14] - first[0][15]
                    + a.fc_0.get_biases()[15] + first[1][14] - b.fc_0.get_biases()[14]
                    - first[1][15] + b.fc_0.get_biases()[15];
    std::copy_n(first[0], 16, joined);
    std::copy_n(first[1], 16, joined + 16);
#if defined(USE_PAIR_ACTIVATIONS)
    a.ac_sqr_0.propagate_pair(joined, h, h + 32);
#else
    a.ac_sqr_0.propagate(joined, h);
    a.ac_0.propagate(joined, h + 32);
#endif
    a.fc_1.propagate(h, second[0]);
    b.fc_1.propagate(h, second[1]);
    std::copy_n(second[0], 16, joined);
    std::copy_n(second[1], 16, joined + 16);
#if defined(USE_PAIR_ACTIVATIONS)
    a.ac_sqr_1.propagate_pair(joined, h + 64, h + 96);
#else
    a.ac_sqr_1.propagate(joined, h + 64);
    a.ac_1.propagate(joined, h + 96);
#endif
    a.fc_2.propagate(h, out[0]);
    b.fc_2.propagate(h, out[1]);
    const i64 merged = (i64(out[0][0]) + out[1][0] + skips) / 2;
    return i32(merged * (600 * OutputScale) / (i64(HiddenOneVal) * (1U << WeightScaleBits) * 2));
}

struct Input {
    alignas(64) std::array<u8, L1> values;
    NNZInfo<L1> nnz;
    Position    pos;
    StateInfo   state;
};
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    Attacks::init();
    Position::init();
    std::mt19937 rng(617);
    auto         experts = std::make_unique<NetworkArchitecture[]>(RoleExperts * 2);
    // Populate ordinary exported matrices through their real parameter loader.
    for (int expert = 0; expert < RoleExperts * 2; ++expert)
    {
        std::ostringstream os(std::ios::binary);
        for (auto [out, in] : {std::pair{16, 1024}, std::pair{16, 64}, std::pair{1, 128}})
        {
            for (int j = 0; j < out; ++j)
                write_little_endian<i32>(os, i32(rng() % 8193) - 4096);
            for (int j = 0; j < out * in; ++j)
                write_little_endian<i8>(os, i8(int(rng() % 255) - 127));
        }
        std::istringstream is(os.str(), std::ios::binary);
        if (!experts[expert].read_parameters(is))
            return 3;
    }
    // Input file is FEN, followed by 1024 already transformed uint8 values per line.
    std::ifstream                       file(argv[1]);
    std::vector<std::unique_ptr<Input>> inputs;
    std::string                         fen, line;
    while (std::getline(file, fen) && std::getline(file, line))
    {
        auto x = std::make_unique<Input>();
        if (x->pos.set(fen, false, &x->state))
            return 4;
        std::istringstream values(line);
        int                v;
        for (auto& item : x->values)
        {
            if (!(values >> v))
                return 5;
            item = u8(v);
        }
        // Architecture-specific NNZ discovery is the same helper used by the FT.
        std::memset(&x->nnz, 0, sizeof(x->nnz));
        for (IndexType i = 0; i < L1; i += 4)
            if (load_as<u32>(x->values.data() + i))
            {
#if defined(USE_AVX512)
                x->nnz.nnz[x->nnz.count++] = i / 4;
#else
                x->nnz.bitset[(i / 4) / 8] |= u8(1u << ((i / 4) % 8));
#endif
            }
        inputs.push_back(std::move(x));
    }
    if (inputs.empty())
        return 6;
    const u8 mask[11] = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 31};
    std::cout << "sizes " << sizeof(NetworkArchitecture) << ' '
              << sizeof(NetworkArchitecture) * RoleExperts * 2 << ' ' << inputs.size() << '\n'
              << std::flush;
    int mode, kind;
    while (std::cin >> mode >> kind)
    {
        i64        checksum = 0;
        const auto start    = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < 250; ++repeat)
            for (std::size_t i = 0; i < inputs.size(); ++i)
            {
                const auto& x     = *inputs[i];
                const auto  color = x.pos.side_to_move();
                // Include extraction, mask lookup and bank dispatch in every case.
                int own = perspective_route(x.pos, color, mask).expert;
                int opp = perspective_route(x.pos, ~color, mask).expert;
                checksum += own + opp;
                if (kind == 0)
                    own = opp = 79;
                else if (kind == 2)
                {
                    own = int((i * 37 + repeat * 13) % RoleExperts);
                    opp = int((i * 19 + repeat * 43) % RoleExperts);
                }
                const auto& a = experts[own];
                const auto& b = experts[RoleExperts + opp];
                checksum += mode == 0 ? a.propagate(x.values.data(), x.nnz, b)
                                      : separate(a, b, x.values.data(), x.nnz);
            }
        const auto elapsed =
          std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
            .count();
        std::cout << mode << ' ' << kind << ' ' << elapsed / (250 * inputs.size()) << ' '
                  << checksum << '\n'
                  << std::flush;
    }
}
