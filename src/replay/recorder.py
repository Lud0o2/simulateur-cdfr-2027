"""Construction des frames d'affichage d'un replay."""


def _frame(simulation, label):
    def point(coords):
        return [round(value, 3) for value in coords]

    movement = simulation.movement
    blocks = simulation.blocks
    return {
        "t": round(simulation.time, 6),
        "position": point(simulation.position),
        "heading": round(simulation.heading, 6),
        "robot": [point(corner) for corner in simulation.footprint],
        "piles": [[point(corner) for corner in block.corners] for block in blocks],
        "cups": [[point(cup.position), cup.enabled] for cup in simulation.suction_cups],
        "target": point(movement.end) if simulation.moving and not movement.turn else None,
        "blocked": bool(movement and (movement.blocked or movement.rotation_blocked)),
        "score": simulation.score_for_blocks(blocks),
        "action": label,
    }


