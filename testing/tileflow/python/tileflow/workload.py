
__all__ = ["GeneralOperator"]


def GeneralOperator(ctx, tensors, loops, levels, fbody):
    for loop in loops:
        assert len(loop) == 2 * levels, f"{len(loop)} vs {2 * levels}"
    org_loops = [x[0] for x in loops]

    def level_operator(lx):
        idx = int((levels - lx) * 2 + 1)
        if lx <= 1:
            with ctx.tile(f"L{lx-1}", [x[idx] for x in loops], "Temporal"):
                fbody(*tensors, *org_loops)
        else:
            with ctx.tile(f"L{lx-1}", [x[idx] for x in loops], "Temporal"):
                with ctx.tile(f"L{lx-1}", [x[idx+1] for x in loops], "Spatial"):
                    level_operator(lx-1)

    level_operator(levels)
