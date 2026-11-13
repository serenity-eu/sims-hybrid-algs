# Publication data

Research data for:

> **Hybrid Exact and Metaheuristic Method for the Multiobjective Satellite
> Image Mosaic Selection Problem**
>
> Vladyslav Hlushchenko<sup>1</sup>, Manuel Combarro Simón<sup>2</sup>,
> Maciej Antczak<sup>1</sup>, Andrzej Jaszkiewicz<sup>1</sup>,
> Jędrzej Musiał<sup>1</sup>
>
> <sup>1</sup> Institute of Computing Science, Poznan University of
> Technology, Piotrowo 3A, 60-965 Poznan, Poland
>
> <sup>2</sup> Interdisciplinary Centre for Security, Reliability and Trust,
> University of Luxembourg, 2 place de l'Université, L-4365
> Esch-sur-Alzette, Luxembourg
>
> Corresponding author: Vladyslav Hlushchenko
> <vladyslav.hlushchenko@cs.put.poznan.pl>

The catalogue snapshot the problem instances were built from, the instances,
the exact fronts that seed the hybrid, and the experiment results the paper's
tables and figures are generated from.

**No satellite imagery is stored here.** This deposit holds catalogue
*metadata* -- footprint, cost, cloud cover, incidence angle -- and the scene id
of every image. The clouds in every instance are synthetic, generated from the
seed recorded in that instance's `.meta.json`. Instance generation reads five
fields per scene: `geometry`, `cost`, `resolution`, `cloudCoverage`,
`providerProperties.incidenceAngle`.

## Pipeline

| stage | artifact | produced by | reproducible |
|---|---|---|---|
| 0 | `satellite-data/scene_catalogue/` | UP42 catalogue search + order estimate | **no** |
| 1 | `satellite-data/aois/` | areas of interest | n/a |
| 2 | `satellite-data/benchmark_instances/` | `sims-preprocessing/generate_benchmark_instances.sh` | **yes, byte-for-byte** |
| 3 | `sims-core/tests/data/{gpbaa,an}_2d_gurobi_hard/` | `sims-problem/generate_pseudo.py` | yes |
| 4 | `biobjective/{gpba,an}/` | `biobjective/scripts/run_all_experiments.sh` | **no** |
| 5 | the paper's tables and figures | `biobjective/scripts/rebuild_paper_artifacts.sh` | **yes, byte-for-byte** |

Stage 0 is a frozen snapshot: the UP42 catalogue is live and `cost` is a live
order estimate, so a search run today returns a different scene set at
different prices. Stage 4 has a 200 s wall-clock budget per run, so a re-run
agrees statistically, not bitwise. Rebuild from the stored results to
reproduce the published figures exactly.

Verify any stage:

```bash
python publication-data/scripts/make_manifest.py --check
```

## Recovering the imagery

Each `<instance>.meta.json` carries `scene_ids`, in the same order as the
image arrays in the matching `.dzn`: image *j* of the instance is
`scene_ids[j]`. Every scene is hosted by `oneatlas`.

```bash
python publication-data/scripts/fetch_quicklooks.py --instance lagos_nigeria_150 --out-dir ./quicklooks
python publication-data/scripts/fetch_quicklooks.py --all --dry-run   # list the 1,109 scenes
```

Quicklooks and thumbnails are previews: free, low resolution, no order. Full
data products require a paid order, and the recorded `cost` is a snapshot of a
live estimate rather than a current price. Nothing in the paper requires the
imagery.

Credentials are read from `UP42_USERNAME` and `UP42_PASSWORD` in the
environment, never passed on the command line.

## Known defect in the archived instances

`incidence_angle` is **0 for every image in all five paris and all five
rio_de_janeiro instances**. Those two cities' pool records store
`providerProperties` as a Python `repr` string rather than JSON, which the
generator cannot parse; it falls back to `0.0`.

No published result is affected: incidence feeds only the `MaxIncidenceAngle`
objective, which this biobjective paper does not use, and it takes no part in
pool filtering or instance selection.

The snapshot records the **effective** value in
`providerProperties.incidenceAngle`, so the instances regenerate byte-identical
to the ones the published results were computed from. The true catalogue value
is preserved alongside it as `incidenceAngleCatalog` wherever the two differ.
**Read `incidenceAngleCatalog` when reusing this dataset for three- or
four-objective work.**

## Layout

```
publication-data/
  MANIFEST.sha256              SHA-256 of all 1,010 archived artifacts, by stage
  scripts/
    fetch_catalogue.py         fresh catalogue pool (not the archived snapshot)
    fetch_quicklooks.py        imagery by recorded scene id
    make_manifest.py           write / --check the manifest
  satellite-data/
    aois/                      5 areas of interest
    scene_catalogue/     catalogue snapshot, 7,684 scenes
    benchmark_instances/         25 instances + .meta.json sidecars
  biobjective/
    gpba/, an/                 25 instances x 17 configurations, + 850 traces
    scripts/                   run_all_experiments.sh, rebuild_paper_artifacts.sh
```

Protocol, hardware and re-run instructions are in the repository root
`README.md`.
