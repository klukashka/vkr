# Self-Organized Codes

*Русская версия (основная): [README.md](README.md)*

Author: Pavel Klyukin, group M8O-407B-23, Moscow Aviation Institute, Department 806.
Supervisor: August Borisov, senior lecturer.

A simulation of information transmission between nodes with distortion, built to answer one question: **can a population spontaneously develop an error-correcting code?**

---

## The problem

When information is passed along, it changes. The receiver does not copy a message — it reconstructs it from what it already holds. Yet messages stay recognizable across many hops.

So the interesting question is not where variation comes from, but where **stability** comes from under constant distortion.

One plausible answer: distortions are not random. They are directed, and roughly in the same direction for different receivers, so messages do not drift apart — they converge on forms that are easy to hold. In existing models these forms are supplied by hand as a parameter. Here they are not supplied at all.

## The connection to coding theory

A stable form does exactly what a codeword does: a corrupted message is pulled back to the nearest admissible one, so errors do not accumulate from hop to hop.

The difference is that a code is designed in advance, while here the set of admissible words has to arise on its own. Such a code, not designed in advance but formed in the population, is called *self-organized* here.

**Hypothesis.** The minimum distance of the self-organized code grows with the channel noise level — the system adapts its redundancy to the transmission conditions.

---

## How it works

Nodes exchange binary vectors of fixed length `n` over a noisy channel. Each node holds a set of weighted records. Nothing about which vectors are "correct" is specified anywhere.

**Receiving.** An incoming vector is matched to the nearest record by Hamming distance. Close enough — the record is reinforced and shifted toward the input by bitwise majority. Too far — a new record is created. This is minimum-distance decoding, except the reference set is accumulated rather than given.

**Sending.** A record is drawn with probability proportional to its weight and transmitted; each bit flips with probability `p`.

**Roles.** Each transmission carries a role label — a plain integer in `[0, R)` with no internal structure. It marks *what* the message is about, separately from *how* it looks. If a receiver assigns an incoming vector to a record belonging to a different role, the record is penalized.

Roles exist because the requirement to keep `R` things distinct is what forces the vectors apart — the same requirement that gives a code its minimum distance. Without roles nothing pushes words apart: in v0.1, with `R = 1` the population keeps a few close synonyms rather than a code.

**Forgetting.** Weights decay each tick; records below threshold are dropped. Capacity is bounded by competition, not by an imposed ceiling.

---

## What gets measured

After a run, all records from all nodes are pooled and clustered by Hamming distance. Cluster centers are the candidate codewords, and their set `C` represents the self-organized code.

| Quantity | Meaning |
|---|---|
| `n` | vector length (fixed) |
| `\|C\|`, `k = log₂\|C\|` | how many distinct words survived |
| `d` | minimum distance between distinct words |
| `d_roles` | minimum distance between words of *different* roles (synonyms ignored) |
| `d_top` | minimum distance of the code made of the heaviest word of each role (exactly `R` words) — the main metric |
| purity | share of a cluster's weight carried by its dominant role |
| error rate | measured directly by pushing `C` through the channel and decoding |

These are then compared against random codes of the same size, the Hamming (sphere-packing) bound and known codes with the same parameters.

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

Each configuration is run with several seeds (30 for the main noise sweep, 10 for the other v0.1 series); results are averaged and spread is reported.

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
git clone https://github.com/klukashka/vkr.git && cd vkr
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# single run; override any SimConfig field with KEY=VALUE
.venv/bin/python -m self_organized_codes run --set noise=0.05 n=12 n_roles=4 --snapshot-every 500

# vary one parameter over several seeds (in parallel), per-run rows to CSV
.venv/bin/python -m self_organized_codes sweep noise 0,0.02,0.05,0.1 --seeds 30 --out results/noise.csv

.venv/bin/pytest
```

Code layout (`self_organized_codes/`):

| Module | Contents |
|---|---|
| `config.py` | `SimConfig` — every parameter, with README symbols in its docstring |
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

---

## Findings (v0.1)

Unless stated otherwise: `n = 12`, `R = 4`, `N = 20` nodes, `θ = 2`, `λ = 0.03`, `role_penalty = 0.1`, 10 000 ticks.
The noise sweep uses 30 seeds, the other series 10.

| `p` | `d_top` | words in `C` | purity |
|---|---|---|---|
| 0 | 3.97 ± 0.85 | 4.7 | 1.00 |
| 0.02 | 4.80 ± 0.55 | 4.7 | 1.00 |
| 0.04 | 5.50 ± 0.68 | 4.3 | 1.00 |
| 0.06 | 4.27 ± 0.87 | 12.8 | 0.76 |
| 0.10 | 4.37 ± 1.07 | 10.5 | 0.59 |

- Without noise `d_top` equals that of a random 4-word code (3.81): distance does not self-organise. Up to `p = 0.04`
  it grows to 5.5 with all roles kept, in line with the hypothesis.
- At `p ≈ 0.05–0.06` the code breaks down: about three times more words, mixed roles, decoding errors of a few
  percent. With a larger `θ` the breakdown comes later.
- Stable regime (one word per role): `θ ≥ 2` and `λ ≤ 0.03`. At `θ = 1` the code disperses or collapses; at
  `λ = 0.1` it fragments into many close words.
- With `R ≥ 8` some roles are lost.
- Not confirmed: with `R = 1` and noise the population does not collapse to one vector; the [7,4] control at `R = 4`
  is uninformative, because four roles need only part of the code.
- Changes made during v0.1: `role_penalty` 0.5 → 0.1 (roles mixed otherwise), runs extended from 3 000 to 10 000
  ticks, `d_top` adopted as the main metric because `d` is lowered by synonyms.

---

## Status

v0.1: the model, measurement and sweep tooling are implemented and tested; the exploratory series in
`experiments/v01.sh` are done. Defaults (`decay = 0.03`, `role_penalty = 0.1`) are tuned for `R = 4`.

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