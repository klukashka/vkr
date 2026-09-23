# Emergent Codes

*Русская версия: [README.ru.md](README.ru.md)*

A simulation of information transmission between nodes with distortion, built to answer one question: **can a population spontaneously develop an error-correcting code?**

---

## The problem

When information is passed along, it changes. The receiver does not copy a message — it reconstructs it from what it already holds. Yet messages stay recognizable across many hops.

So the interesting question is not where variation comes from, but where **stability** comes from under constant distortion.

One plausible answer: distortions are not random. They are directed, and roughly in the same direction for different receivers, so messages do not drift apart — they converge on forms that are easy to hold. In existing models these forms are supplied by hand as a parameter. Here they are not supplied at all.

## The connection to coding theory

A stable form does exactly what a codeword does: a corrupted message is pulled back to the nearest admissible one, so errors do not accumulate from hop to hop.

The difference is that a code is designed in advance, while here the set of admissible words has to arise on its own.

**Hypothesis.** The minimum distance of the emergent set grows with the channel noise level — the system adapts its redundancy to the transmission conditions.

---

## How it works

Nodes exchange binary vectors of fixed length `n` over a noisy channel. Each node holds a set of weighted records. Nothing about which vectors are "correct" is specified anywhere.

**Receiving.** An incoming vector is matched to the nearest record by Hamming distance. Close enough — the record is reinforced and shifted toward the input by bitwise majority. Too far — a new record is created. This is minimum-distance decoding, except the reference set is accumulated rather than given.

**Sending.** A record is drawn with probability proportional to its weight and transmitted; each bit flips with probability `p`.

**Roles.** Each transmission carries a role label — a plain integer in `[0, R)` with no internal structure. It marks *what* the message is about, separately from *how* it looks. If a receiver assigns an incoming vector to a record belonging to a different role, the record is penalized.

Roles exist because without them the system collapses: merging pulls everything toward a single vector, which is perfectly stable and completely useless. The requirement to keep `R` things distinct is what forces the vectors apart — the same requirement that gives a code its minimum distance.

**Forgetting.** Weights decay each tick; records below threshold are dropped. Capacity is bounded by competition, not by an imposed ceiling.

---

## What gets measured

After a run, all records from all nodes are pooled and clustered by Hamming distance. Cluster centers are the candidate codewords, and their set `C` is the emergent code.

| Quantity | Meaning |
|---|---|
| `n` | vector length (fixed) |
| `\|C\|`, `k = log₂\|C\|` | how many distinct words survived |
| `d` | minimum distance between distinct words |
| `d_roles` | minimum distance between words of *different* roles (synonyms ignored) |
| `d_top` | minimum distance of the code made of the heaviest word of each role (exactly `R` words) — the main metric |
| purity | share of a cluster's weight carried by its dominant role |
| error rate | measured directly by pushing `C` through the channel and decoding |

These are then compared against the Hamming (sphere-packing) and Singleton bounds, and against known codes with the same parameters.

---

## Experiments

| # | Varied | Question |
|---|---|---|
| 1 | `θ`, `λ` | where are the boundaries between collapse, stability and dispersal? |
| 2 | **`p`** | **does `d` grow with channel noise?** (main result) |
| 3 | `R` at fixed `n` | at what point does the system stop coping, and how does that compare to the Hamming bound? |
| 4 | `N`, `L` | is the result robust to population size and turnover? |
| 5 | `p = 0` | does anything new arise without noise? (control) |
| 6 | seeded with Hamming [7,4] | does the dynamics preserve a good code? (control) |

Each configuration is run at least 30 times with different seeds; results are averaged and spread is reported.

---

## Design constraints

- **Binary vectors, Hamming distance.** Classical bounds are defined in these terms; anything else would not be comparable.
- **No admissible set anywhere in the code.** If it were specified, the result would be planted rather than obtained.
- **Noise in the channel.** Without it "error-correcting" means nothing.
- **Random initialization.** The poorer the starting point, the stronger the result. Empty stores and seeded codes are used as controls only.
- **No neural networks.** Every observed effect has to be traceable to a cause, which rules out learned black boxes.

---

## Running

Python 3.11+, NumPy only.

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# single run; override any SimConfig field with KEY=VALUE
.venv/bin/python -m emergent_codes run --set noise=0.05 n=12 n_roles=4 --snapshot-every 500

# vary one parameter over several seeds (in parallel), per-run rows to CSV
.venv/bin/python -m emergent_codes sweep noise 0,0.02,0.05,0.1 --seeds 30 --out results/noise.csv

.venv/bin/pytest
```

Code layout (`emergent_codes/`):

| Module | Contents |
|---|---|
| `config.py` | `SimConfig` — every parameter, with README symbols in comments |
| `channel.py` | binary symmetric channel |
| `node.py` | record store: receive (min-distance decoding), choose, forget |
| `simulation.py` | population dynamics, init modes (`random`, `empty`, `hamming74`), turnover |
| `analysis.py` | pooling and clustering records into `C`; `\|C\|`, `k`, `d`, purity, decoding error, bounds |
| `codes.py` | Hamming distance, Hamming / Singleton bounds, Hamming [7,4] |
| `cli.py` | `run` and `sweep` commands |

Choices made in v0.1 that the text above leaves open:

- the speaker picks a role uniformly, then a record of that role by weight; with no record for the role it invents a random vector and keeps it;
- on a different-role match the record's weight is multiplied by `role_penalty` **and** the input is stored as a new record of the correct role;
- newcomers under turnover start empty;
- clusters holding less than 2% of the pooled weight are treated as noise, not codewords.

Experiment scripts (`experiments/`, raw results go to the gitignored `results/`):

| Script | Series |
|---|---|
| `v01.sh` | noise sweep, `θ × λ` map, number of roles, [7,4] control, noise sweep at larger `θ` |
| `v02.py` | generality over `n` and `R`, `R = 1` long runs, [7,4] control at `R = 16`, `d` ceiling analysis |
| `v02_checks.py` | [7,4] control with `λ` scaled to `R`; generality with `θ` scaled to `n` |
| `v03.py` | number of roles with `λ ∝ 1/R`; `n = 16…24` with 30 seeds |

---

## Findings so far (v0.1–v0.3)

Unless stated otherwise: `N = 20` nodes, 10 000 ticks, `role_penalty = 0.1`, `λ = 0.12 / R` (0.03 at `R = 4`).
Numbers are mean ± sd over seeds; "significant" means the slope of `d_top` on `p` has `|t| ≥ 3`. These are
exploratory series: 10 seeds each unless 30 are stated, below the 30-seed standard set for final results.

**1. In the densely packed regime, noise increases the minimum distance.** At `n = 12, R = 4, θ = 2` (30 seeds)
`d_top` grows from 4.0 ± 0.85 at `p = 0` — no better than a random 4-word code (3.8) — to 5.5 ± 0.7 at `p = 0.04`,
with all roles present and purity ≈ 1. The slope is significant for `n = 8` (`R = 2, 4`), `n = 12` (`R = 2, 4`) and
`n = 16` (`R = 4`: t = 4.7, `R = 8`: t = 4.5, 30 seeds each); it is weaker at `n = 12, R = 8` (t = 2.2) and absent at
`n = 8, R = 8`, where the system is overloaded (purity 0.64).

**2. It is not a general law.** The effect disappears when random words are already far apart relative to `θ`:
`n = 20` is borderline (t ≈ 2), `n = 24` shows nothing (30 seeds), and `R = 2` at `n ≥ 16` never does. Refined
hypothesis: *`d` grows with `p` while the code is densely packed (random `d` within roughly `θ + 2…4`) and noise
stays below the decoding radius.*

**3. Too much noise destroys the code instead of improving it.** Breakdown is set by the expected number of
flipped bits `n·p` relative to `θ`, not by `p` alone: at `n = 12, θ = 2` it happens at `p ≈ 0.05–0.06` (the code
splits into 10–13 mixed-role words, purity 0.5–0.75); at `θ = 3` it moves to `p ≈ 0.07`; at `n = 20–24` with `R ≥ 4`
roles are lost already at `p = 0.04`. Consequently the useful noise window narrows as `n` grows.

**4. `d` stalls well below the optimum.** At `n = 12, R = 4` `d_top` settles at 5.6–5.8 by ~10k ticks and does not
move through 50k ticks for any `θ ∈ {2, 3, 4}`; the optimum is 8 (Plotkin, `A(12, 8) = 4`). The code looks like a
*random code with an exclusion zone*: random 4-word codes conditioned on `d ≥ 5` / `d ≥ 6` average 5.15 / 6.01.
Two causes:
- repulsion is short-range: the chance that a noisy copy lands within `θ` of another role's word falls 15–20× per
  unit of distance (at `θ = 2, d = 6, p = 0.04` it is 3·10⁻⁵ per message);
- words are nearly immobile: at `θ = 4` the force at `d = 6` is still 1.7·10⁻² and `d` does not grow either,
  because a heavy record's majority vector hardly moves and new variants only appear as noisy copies next to it.

**5. Capacity in roles falls with noise, and is far below the bounds.** At `n = 12, θ = 2` with `λ ∝ 1/R` all
16 roles survive at `p = 0.02` (purity 0.99; about 20 of 32 at `R = 32`), but only ~6 at `p = 0.05`. In the dense
regime `d_top` is pinned at `θ + 1 = 3`. For comparison the Hamming bound at `n = 12, d = 3` is 315 words.

**6. Forgetting must scale with the number of roles.** Each record is reinforced about once per `R` messages, so
with a fixed `λ` large `R` is lost to forgetting rather than to noise: a node that forgets a role invents a random
vector for it. With `λ = 0.03` the [7,4] code seeded at `R = 16` fell apart even at `p = 0`, and the apparent
capacity limit of ~8 roles in v0.1 was this artefact. The relevant dimensionless quantities are `λ·R` and `n·p / θ`.

**7. Controls.**
- *Hamming [7,4], `R = 16`, `θ = 1`, `λ = 0.003`:* the seeded code survives intact (16/16 words on their roles,
  `d = 3`) at `p ≤ 0.02` and almost intact at `p = 0.05` (15.4/16). A random start under the same parameters never
  reaches `d = 3` (`d_top` 1.6–2.0). The dynamics preserves a good code but does not find a tight one.
- *No roles (`R = 1`):* the population does *not* collapse to one vector when there is noise. Over 60k ticks at
  `p = 0` it slowly coarsens (4.6 → 1.6 words, top word's share 0.41 → 0.85), but at `p = 0.05` diversity is
  stationary (≈5 words, top share ≈0.45, 15% of messages create records): a mutation–drift balance. The result is
  synonyms without separation (`d ≈ 3`), not a code. Roadmap step 2 holds only in the noiseless limit.

**Open questions.** What limits capacity at ~20 roles at `p = 0.02`, given that geometry allows far more? Would a
mechanism for words to move (rather than only be selected) break the exclusion-zone ceiling, and does adding it
stay within the design constraints above?

---

## Status

v0.3: the model, measurement and sweep tooling are implemented and tested; the exploratory series above are done.
Defaults (`decay = 0.03`, `role_penalty = 0.1`) are tuned for `R = 4`; for other `R` use `decay = 0.12 / R`.

Roadmap:

1. Single node, closed on itself
2. Population without roles — confirm that it degenerates
3. Roles enabled; map the parameter region where the repertoire survives
4. Noise sweep — the main result
5. Controls
6. Comparison against bounds and known codes

---

## Out of scope

Deliberately left out, kept as directions for further work: the environment as a node with its own parameters, links between records and spreading activation, consolidation and reflection, modelling the addressee when choosing what to send, variable edge weights, representations based on compression or embeddings.

None of them serve the question being asked, and adding them turns the project back into architecture-building.