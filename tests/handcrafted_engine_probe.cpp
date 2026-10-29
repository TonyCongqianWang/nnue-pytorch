// CPU-only integration probe. Link against the matching engine objects except main.o.
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
    Attacks::init();
    Position::init();
    if (argc == 2 && std::string(argv[1]) == "--material")
    {
        for (int x : NominalToCompact)
            std::cout << x << ' ';
        std::cout << '\n';
        int p, n, b, r, q, l;
        while (std::cin >> p >> n >> b >> r >> q >> l)
            std::cout << material_bucket(p, n, b, r, q, l != 2, l == 0) << '\n';
        return 0;
    }
    if (argc != 2)
        return 2;
    auto     net = std::make_unique<Network>();
    EvalFile file;
    net->load("", argv[1], file);
    if (!file.current)
        return 3;
    auto        cache      = std::make_unique<AccumulatorCaches>(*net);
    auto        freshCache = std::make_unique<AccumulatorCaches>(*net);
    auto        acc        = std::make_unique<AccumulatorStack>();
    auto        fresh      = std::make_unique<AccumulatorStack>();
    std::string fen, moves;
    int         checked = 0;
    while (std::getline(std::cin, fen) && std::getline(std::cin, moves))
    {
        Position  pos;
        StateInfo states[MAX_PLY + 1];
        if (pos.set(fen, false, &states[0]))
        {
            std::cout << "rejected\n";
            continue;
        }
        acc->reset();
        std::vector<Move> played;
        auto              check = [&]() {
            fresh->reset();
            freshCache->clear(*net);
            const auto incremental = net->evaluate(pos, *acc, *cache);
            const auto rebuilt     = net->evaluate(pos, *fresh, *freshCache);
            const auto trace       = net->trace_evaluate(pos, *fresh, *freshCache);
            if (incremental != rebuilt || trace.positional != incremental
                || acc->latest().accumulation != fresh->latest().accumulation)
            {
                std::cerr << "Mismatch " << pos.fen() << '\n';
                return false;
            }
            ++checked;
            return true;
        };
        if (!check())
            return 5;
        const auto t = net->trace_evaluate(pos, *fresh, *freshCache);
        std::cout << "root " << t.stm.bucket << ' ' << t.stm.expert << ' ' << t.opp.bucket << ' '
                  << t.opp.expert << ' ' << t.positional << '\n';
        std::istringstream input(moves);
        std::string        token;
        while (input >> token)
        {
            if (played.size() >= MAX_PLY)
                return 6;
            if (token == "0000")
            {
                if (pos.checkers())
                    return 6;
                played.push_back(Move::null());
                pos.do_null_move(states[played.size()]);
                // Null moves preserve accumulators; side-to-move changes role selection.
                if (!check())
                    return 7;
                continue;
            }
            const Move m = UCIEngine::to_move(pos, token);
            if (m == Move::none())
                return 6;
            played.push_back(m);
            pos.do_move(m, states[played.size()], pos.gives_check(m), acc->push(pos));
            if (played.size() % 3 == 0 && !check())
                return 7;
        }
        if (!check())
            return 8;
        while (!played.empty())
        {
            if (played.back() == Move::null())
                pos.undo_null_move();
            else
            {
                pos.undo_move(played.back());
                acc->pop();
            }
            played.pop_back();
            if (played.size() % 2 == 0 && !check())
                return 9;
        }
    }
    std::cout << "verified " << checked << '\n';
}
