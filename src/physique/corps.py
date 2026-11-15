"""Corps physiques internes : piles et prises."""

from dataclasses import dataclass

import pymunk

from ..domaine.modele import Block, Point
from .collisions import _world_corners


@dataclass(frozen=True)
class _Grip:
    cup: int
    shape: pymunk.Poly
    angle: float


@dataclass
class _Pile:
    body: pymunk.Body
    shape: pymunk.Poly
    floor: tuple[pymunk.Constraint, pymunk.Constraint]
    size: Point
    grip: _Grip | None = None
    _block: Block | None = None

    def free_block(self) -> Block:
        position, angle = tuple(self.body.position), self.body.angle
        if self._block is None or self._block.position != position or self._block.angle != angle:
            self._block = Block(position, angle, _world_corners(self.shape))
        return self._block
