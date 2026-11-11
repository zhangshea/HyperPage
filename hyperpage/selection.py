"""Selection on validation replays (hot-record share, deployed layout), savings over the incumbent, and their tests."""
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

# Replay results are dictionaries of rows keyed 'layout|draw|budget|share'; a row holds the page reads of every step.


def row_key(layout, draw, budget, share):
    return f'{layout}|{draw}|{budget}|{share}'


def scored_reads(rows, layout, draw, budget, share, warmup, scored):
    """Validation reads of one row over the scored steps: `scored` steps after a warm-up of `warmup` steps."""
    return float(np.sum(rows[row_key(layout, draw, budget, share)]['per_step'][warmup:warmup + scored]))


def validation_share(rows, layout, draws, budget, shares, warmup=10, scored=20):
    """Hot-record share of one layout: fewest validation reads averaged over the draws, ties to the smaller share."""
    # [Paper] Section VII-A, Metrics: each layout gets the share with the fewest validation reads after a warm-up
    total = {}
    for share in shares:
        per_step = np.mean([rows[row_key(layout, k, budget, share)]['per_step'] for k in draws], axis=0)
        total[share] = float(per_step[warmup:warmup + scored].sum())
    return min(shares, key=lambda s: (total[s], s))


def validation_share_of_draw(rows, layout, draw, budget, shares, warmup=10, scored=20):
    """Hot-record share of one layout chosen on one draw, ties to the smaller share."""
    return min(shares, key=lambda s: (scored_reads(rows, layout, draw, budget, s, warmup, scored), s))


def deployment(rows, candidates, incumbent, draw, budget, shares, warmup=10, scored=20):
    """HyperPage's selection for one draw and budget: the (layout, share) pair with the fewest validation reads.

    The incumbent with every share is always a candidate. Ties go to the incumbent, then to the smaller share, then
    to the candidate listed first. Returns (layout, share, validation reads).
    """
    # [Paper] Section VI-D: every candidate layout, the incumbent included, is replayed for every share of the grid
    options = [(scored_reads(rows, incumbent, draw, budget, s, warmup, scored), 0, s, -1, incumbent) for s in shares]
    for position, layout in enumerate(candidates):
        if layout == incumbent:
            continue
        options += [(scored_reads(rows, layout, draw, budget, s, warmup, scored), 1, s, position, layout) for s in shares]
    best = min(options)
    return best[4], best[2], best[0]


def total_reads(rows, layout, draw, budget, share):
    return float(np.sum(rows[row_key(layout, draw, budget, share)]['per_step']))


def pooled_saving(layout_totals, incumbent_totals):
    """Saving over the incumbent in percent, pooled over the draws: 100 * (1 - sum_k R_k / sum_k R_incumbent,k)."""
    # [Paper] Section VII-A, Metrics: M = 1 - R_A / R_MARGO for total reads R averaged over the three training draws
    return 100.0 * (1.0 - float(np.sum(layout_totals)) / float(np.sum(incumbent_totals)))


def saving_per_draw(layout_totals, incumbent_totals):
    return [100.0 * (1.0 - a / b) for a, b in zip(layout_totals, incumbent_totals)]


def round_half_up(value, places=1):
    """Decimal rounding with halves away from zero, as printed in the tables."""
    return float(Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def block_bootstrap(layout_steps, incumbent_steps, block, resamples=2000, seed=0x13528f0, level=0.05):
    """Paired moving-block bootstrap of the saving over the steps.

    layout_steps and incumbent_steps are the reads per step averaged over the draws. Blocks of `block` consecutive
    steps are resampled with replacement and both series use the same blocks. Returns (low, high, differs): the
    level/2 and 1 - level/2 percentiles of the resampled saving in percent, and whether they exclude zero.
    """
    # [Paper] Section VII-A: paired bootstrap over the test steps with 2,000 resamples, also with moving blocks
    a = np.asarray(layout_steps, float)
    b = np.asarray(incumbent_steps, float)
    n = len(a)
    assert n == len(b)
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(resamples, n_blocks))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(resamples, -1)[:, :n]
    saving = 100.0 * (1.0 - a[index].sum(1) / b[index].sum(1))
    low, high = np.percentile(saving, [100 * level / 2, 100 * (1 - level / 2)])
    return float(low), float(high), bool(low > 0 or high < 0)


def budget_bracket(incumbent_reads_by_budget, target_reads, declared):
    """Budget at which the incumbent reads as few pages as a target, as a bracket on a budget grid.

    incumbent_reads_by_budget maps a grid budget (same unit as declared) to the incumbent's test reads at its own
    validation share. Returns (previous / declared, first / declared), where first is the smallest grid budget not
    below the declared one at which the incumbent reads at most target_reads, and previous the grid point before it.
    """
    # [Paper] Table III, last row
    grid = sorted(incumbent_reads_by_budget)
    reached = [g for g in grid if g >= declared - 1e-9 and incumbent_reads_by_budget[g] <= target_reads]
    first = reached[0]
    previous = max(g for g in grid if g < first - 1e-12)
    return previous / declared, first / declared
