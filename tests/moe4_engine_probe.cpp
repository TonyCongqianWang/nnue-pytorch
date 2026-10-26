// Compare Stockfish's incremental accumulators with fresh reconstruction.
// Link against a compatible Stockfish build, excluding main.o and obsolete feature objects.
// stdin: alternating FEN and whitespace-separated UCI move sequence lines.
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "attacks.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "position.h"
#include "uci.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    Attacks::init();
    Position::init();
    auto     net = std::make_unique<Network>();
    EvalFile file;
    net->load("", argv[1], file);
    if (!file.current)
        return 3;
    auto        cache      = std::make_unique<AccumulatorCaches>(*net);
    auto        freshCache = std::make_unique<AccumulatorCaches>(*net);
    auto        acc        = std::make_unique<AccumulatorStack>();
    auto        fresh      = std::make_unique<AccumulatorStack>();
    int         checked    = 0;
    std::string fen, moves;
    while (std::getline(std::cin, fen) && std::getline(std::cin, moves))
    {
        Position  pos;
        StateInfo states[MAX_PLY + 1];
        if (pos.set(fen, false, &states[0]))
            return 4;
        acc->reset();
        std::vector<Move> played;
        auto              check = [&] {
            fresh->reset();
            freshCache->clear(*net);
            auto incremental = net->evaluate(pos, *acc, *cache);
            auto rebuilt     = net->evaluate(pos, *fresh, *freshCache);
            if (incremental != rebuilt
                || acc->latest().accumulation != fresh->latest().accumulation)
            {
                std::cerr << "Accumulator mismatch at " << pos.fen() << '\n';
                return false;
            }
            auto trace = net->trace_evaluate(pos, *fresh, *freshCache);
            if (trace.positional[0] != incremental || trace.positional[0] != trace.positional[3])
            {
                std::cerr << "Trace/symmetry mismatch at " << pos.fen() << '\n';
                return false;
            }
            ++checked;
            return true;
        };
        if (!check())
            return 5;
        std::istringstream input(moves);
        std::string        token;
        while (input >> token)
        {
            const Move move = UCIEngine::to_move(pos, token);
            if (move == Move::none())
                return 6;
            played.push_back(move);
            pos.do_move(move, states[played.size()], pos.gives_check(move), acc->push());
            // Exercise lazy forward/backward reconstruction as well as immediate updates.
            if (played.size() % 3 == 0 && !check())
                return 7;
        }
        if (!check())
            return 8;
        while (!played.empty())
        {
            pos.undo_move(played.back());
            acc->pop();
            played.pop_back();
            if (played.size() % 2 == 0 && !check())
                return 9;
        }
    }
    std::cout << "Verified " << checked << " incremental/fresh positions and trace symmetry\n";
}
