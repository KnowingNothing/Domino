
__all__ = ["GeneralOperator"]


def GeneralOperator(ctx, tensors, loops, levels, fbody):
    for loop in loops:
        if loop is not None:
            assert len(loop) == 2 * levels + 2, f"{len(loop)} vs {2 * levels + 2}"
    
    # org_loops = [None if x is None else x[0] for x in loops]
    org_loops = []
    for x in loops:
        if x is None:
            org_loops.append(None)
        else:
            org_loops.append(x[0])
    def level_operator(lx):
        idx = int((levels - lx) * 2 + 1)
        if lx <= 0:
            cur_loops = []
            for x in loops:
                if x is not None:
                    cur_loops.append(x[idx])
            with ctx.tile(f"L{lx}", cur_loops, "Temporal"):
                # with ctx.tile(f"L{lx-1}", [x[idx+1] for x in loops], "Temporal"):
                fbody(*tensors, *org_loops)
        else:
            cur_loops = []
            next_loops = []
            for x in loops:
                if x is not None:
                    cur_loops.append(x[idx])
                    next_loops.append(x[idx+1])
            with ctx.tile(f"L{lx}", cur_loops, "Temporal"):
                with ctx.tile(f"L{lx}", next_loops, "Spatial"):
                    level_operator(lx-1)
    level_operator(levels)
