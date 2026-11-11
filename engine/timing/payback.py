"""Payback of a timed layout: how many later runs repay its refinement and the index rewrite, in wall time and in
core-seconds, from the median seconds per step of the incumbent and the median paired saving (summarize.py)."""
# [Paper] Section VII-B, Search time: "Payback charges the refinement in core-seconds and the index rewrite. With 400
# steps per run ...". saved = s_per_step(incumbent) x median saving; wall steps = (prep + refine + rewrite) / saved;
# runs of wall time = wall steps / steps per run; runs in core-seconds = wall steps / cores / steps per run.
# The investment is a number of core-seconds in both figures: the wall figure divides it by the wall seconds that
# the 12-thread engine saves per step, the core-second figure by the core-seconds it saves per step.
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('summary', help='JSON output of summarize.py')
    ap.add_argument('--compare', default='HP:MARGO', help='pair of arms of the summary, layout:incumbent')
    ap.add_argument('--investment', type=float, required=True,
                    help='single-thread seconds to prepare and refine the layout '
                         '(multi-thread steps charged per thread)')
    ap.add_argument('--rewrite', type=float, required=True, help='seconds to write the index in the new layout')
    ap.add_argument('--steps-per-run', type=int, default=400, help='steps of one training run')
    ap.add_argument('--cores', type=int, default=12, help='threads of the timed engine')
    a = ap.parse_args()
    o = json.load(open(a.summary))['pairs'][a.compare]
    A, B = a.compare.split(':')
    saved = o['s_per_step'][B] * o['median_saving'] / 100          # wall seconds saved per step
    inv = a.investment + a.rewrite
    wall_steps = inv / saved if saved > 0 else float('inf')
    res = dict(compare=a.compare, saved_wall_s_per_step=saved, investment_s=inv, payback_steps_wall=wall_steps,
               payback_steps_core=wall_steps / a.cores, runs_wall=wall_steps / a.steps_per_run,
               runs_core=wall_steps / a.cores / a.steps_per_run)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
