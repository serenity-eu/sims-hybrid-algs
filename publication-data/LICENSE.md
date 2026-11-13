# Licensing and attribution

This deposit contains material under three different terms. They are listed
separately because one licence cannot cover all of it.

## 1. Code — MIT

Everything under `scripts/` and `biobjective/scripts/`, and the repository's
source code, are covered by the MIT licence in the repository root.

## 2. Derived research outputs — CC BY 4.0

Creative Commons Attribution 4.0 International
(<https://creativecommons.org/licenses/by/4.0/>), © 2026 Poznan University of
Technology.

This covers the work produced by the authors:

- `satellite-data/benchmark_instances/` — the 25 problem instances. Cloud
  cover in these instances is **synthetic**, generated from the recorded
  seed; the geometry is the authors' own fragment discretisation of the
  areas of interest.
- `satellite-data/aois/` — the areas of interest.
- `biobjective/` — all experiment results and binary traces.
- `MANIFEST.sha256`, `README.md`.

## 3. Satellite catalogue metadata — third-party

`satellite-data/scene_catalogue/` is **not** the authors' work and is **not**
covered by the licences above.

It is a 2,491-record extract of catalogue metadata retrieved through the UP42
platform: scene identifier, footprint, acquisition date, cloud cover,
resolution, incidence angle, and the order-estimate price recorded at
retrieval time. It contains **no imagery** — no pixels of any kind are
distributed in this deposit.

Imagery copyright, by collection:

| collection | scenes | attribution |
|---|---|---|
| `phr` (Pléiades) | 1,500 | © CNES (year of acquisition), Distribution Airbus DS |
| `pneo` (Pléiades Neo) | 966 | © Airbus DS (year of acquisition) |
| `spot` (SPOT) | 25 | © Airbus DS (year of acquisition) |

Acquisition years span 2021–2026; each record carries its own
`acquisitionDate`.

The extract is redistributed for the sole purpose of allowing the published
results to be reproduced and verified, as required by the journal's research
data policy. Recorded prices are historical order estimates at retrieval
time, are not current prices, and are not an offer.

Re-use of this metadata beyond reproducing the published results is governed
by the terms of Airbus Defence and Space and the UP42 platform, not by this
deposit. Users intending any other use should obtain their own licence.

**Removal.** If Airbus Defence and Space or UP42 object to the redistribution
of this extract, contact the corresponding author and it will be removed from
the deposit. The remaining material is unaffected: the instances, results and
figures are self-contained, and every artifact's checksum is recorded in
`MANIFEST.sha256`, so the published results remain verifiable without it.
