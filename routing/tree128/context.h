// Shared lazy feature extraction. Adapter API: count(side, P..K = 0..5),
// occupied(side), pieces(side, type), king(side). Side 0 is side to move.
#pragma once
#include <algorithm>
#include <cstdint>
namespace frozen_tree128 {
template<class Board> class Context {
    const Board board;
    mutable int values[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    static int relative(int total, int pawns) {
        int p = std::min({8, total, pawns});
        return !p ? 0 : p <= total-p ? 1 : p <= total-p+2 ? 2 : 3;
    }
    int wing(int side) const {
        auto p = board.pieces(side, 0);
        return bool(p & 0x0f0f0f0f0f0f0f0fULL) + 2*bool(p & 0xf0f0f0f0f0f0f0f0ULL);
    }
    int compute(int i) const {
        if (i < 6) {
            int side = i/3;
            if (i%3 == 0) return std::min(15, board.occupied(side)-1);
            if (i%3 == 1) return relative(get(side*3), board.count(side, 0));
            int n=board.count(side, 1), b=board.count(side, 2);
            int r=board.count(side, 3), q=board.count(side, 4);
            auto bishops = board.pieces(side, 2);
            int pair = bool(bishops & 0x55aa55aa55aa55aaULL) && bool(bishops & 0xaa55aa55aa55aa55ULL);
            return 2*(!q ? (!r ? 2 : 1) : 0) + int(n+b+pair > r+2*q);
        }
        if (i == 6) {
            int us=board.king(0), them=board.king(1);
            int ur=(us/8) ^ (board.white_to_move() ? 0 : 7);
            int tr=(them/8) ^ (board.white_to_move() ? 7 : 0);
            return 4*int(ur>=2) + 2*int(ur>tr) + int((us%8<4)!=(them%8<4));
        }
        int us=wing(0), them=wing(1);
        return us==3 && them==3 ? 0 : us==3 ? 1 : them==3 ? 2 : us && them && us!=them ? 3 : 4;
    }
public:
    explicit Context(Board b): board(b) {}
    int get(int i) const {
        if (values[i] < 0) values[i] = compute(i);
        return values[i];
    }
};
}
