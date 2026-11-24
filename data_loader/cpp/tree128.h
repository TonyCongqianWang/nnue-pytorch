#pragma once
#include "lib/chess.h"
#include "tree128_generated.h"
#include "../../routing/tree128/context.h"
namespace tree128_loader {
using namespace chess;
struct Tree128Board {
    const Position& pos;
    Color color(int side) const { return side ? !pos.sideToMove() : pos.sideToMove(); }
    bool white_to_move() const { return pos.sideToMove() == Color::White; }
    int occupied(int side) const { return pos.piecesBB(color(side)).count(); }
    int count(int side, int type) const { return pos.piecesBB(Piece(PieceType(type), color(side))).count(); }
    std::uint64_t pieces(int side, int type) const { return pos.piecesBB(Piece(PieceType(type), color(side))).bits(); }
    int king(int side) const { return int(pos.kingSquare(color(side))); }
};

inline int tree128(const Position& pos) {
    return frozen_tree128::select(frozen_tree128::Context<Tree128Board>(Tree128Board{pos}));
}

}
