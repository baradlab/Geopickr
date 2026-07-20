# Changelog

All notable changes to **Geopickr** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project uses
[semantic versioning](https://semver.org/) (with PEP 440 pre-release suffixes,
e.g. `1.1.0b2`).

## [Unreleased]

### Fixed
- **STOPGAP `.star` halfset labels** — the `_halfset` column is now written as the
  string `A`/`B` (alternating) instead of numeric `1`/`2`. STOPGAP's motivelist
  defines `halfset` as a string field valued `A`/`B` (see
  `sg_motl_assign_halfsets.m`); the numeric labels were read as the strings
  `"1"`/`"2"` and never matched, breaking the gold-standard half-set split.

### Changed
- **Even surface picking** — surface sampling now uses a restricted Lloyd
  relaxation (centroidal Voronoi tessellation) over the mesh instead of
  area-weighted random + greedy Poisson thinning. Particles land in an even,
  near-hexagonal density (much lower spread of nearest-neighbor distances) at the
  requested `area / tangential²` count. A short min-distance repulsion pass
  afterwards clears the close-pair tail the chord-metric CVT would otherwise
  leave on curved regions.

### Added
- **Jitter** (surface style) — an optional **Jitter (voxels)** field that
  randomly perturbs each surface particle in the surface plane after the even
  layout is computed, so users who want to break up the regular lattice can.
  Also exposed as `pickparticle ... jitter <voxels>`.

## [1.1.0b3] — 2026-07-16

### Added
- **Pick with an offset** — the Geometry Picker has an **Offset (voxels)** field
  that moves particles along their own +Z axis (surface normal for
  sphere/tube/surface, axis tangent for filament) as they are picked, baked into
  the coordinates. **Show fit** previews the result for sphere/tube as a cyan
  "particle shell" at radius +offset (outside the yellow fit for positive,
  inside for negative). Also exposed as `pickparticle ... offset <voxels>`.
- **Bake the Place Object Z/phi offset into exports** — the Export dialog and
  `geopickr export` gain an "apply display offset" option (default on), so a
  Z/phi offset dialed in for viewing can be folded into the exported
  coordinates. Lets you tweak the offset after picking without re-picking.

### Fixed
- **STOPGAP `.star` compatibility** — the exported loop header now uses bare
  column tags (dropped the RELION-style `#1`, `#2`, … suffixes, which STOPGAP's
  reader does not expect) and includes a blank line between the header and the
  data rows. Matches STOPGAP's own `stopgap_star_write.m`; the missing blank line
  would otherwise cause STOPGAP's reader to drop the first particle.

## [1.1.0b2] — 2026-07-08

Incorporates the ChimeraX RBVI Toolshed review feedback and UI-density work. Beta.

### Added
- **Help** button at the bottom of the tool (opens the documentation) so the
  help is discoverable without the right-click context menu.
- Geopickr **icon / favicon** (a pastel particle-sphere), shown by the README
  and available for the Toolshed listing.

### Changed
- The tool now opens as its **own floating window** (instead of docked) with
  tightened control spacing, so it fits smaller / laptop screens.
- **Denser 2-column layouts** for the Object, Sampling and Utilities sections
  (paired controls share rows).
- **US spelling** throughout the UI, documentation and code comments.
- Documentation: "how to open" now points to the **Geopickr** menu entry and
  its tabs.

### Fixed
- `pickparticle` now raises a clear error ("No markers were specified" /
  "No surface was specified") instead of an internal `IndexError`/traceback when
  the marker or surface model is missing.

## [1.1.0b1] — 2026-06-19

Adds multi-format particle export. Beta.

### Added
- Export to **Dynamo** `.tbl` (with an optional `.vll` volume-list `> table`
  reference), **RELION 5.1** (centered-Ångström) and **RELION 3/4** (pixel) star
  files, alongside the existing `.em` and STOPGAP `.star`.
- **Export…** button (Place Object and Geometry Picker tabs) and a
  `geopickr export` command. Coordinates are converted from ChimeraX scene units
  using a chosen tomogram **Volume** (voxel size / box / origin).
- Dynamo/RELION Euler-angle conventions are **cross-validated against ArtiaX**
  (exported angles reproduce the same orientation through ArtiaX's own rotation
  classes).

### Fixed
- Place Object: the **first** opened motive list now has editable visualization
  controls immediately (previously locked until a second list was opened).
  ([#3](https://github.com/baradlab/Geopickr/issues/3))

### Notes
- The Dynamo/RELION **angle** conventions are ArtiaX-validated; the
  **coordinate-origin** conventions (Dynamo 1-indexing, RELION 5 box/2 centering)
  are still pending confirmation against a live Dynamo/RELION import
  ([#1](https://github.com/baradlab/Geopickr/issues/1),
  [#2](https://github.com/baradlab/Geopickr/issues/2)).

## [1.0.0] — 2026-06-18

First public release. A fork and port of the UCSF Chimera **Place Object**
(2.1.0) and **Pick Particle** (2.0.0) plugins by **Kun Qu** in the **Briggs
laboratory**, integrated and extended for ChimeraX.

### Added
- A single tabbed tool with **Place Points**, **Geometry Picker** and
  **Place Object** tabs.
- Geometric particle picking on **spheres, tubes, filaments and arbitrary
  surface meshes**, with live per-object radius fitting.
- Efficient **instanced rendering** (one `Surface` + `Places` per motive list),
  scaling to tens of thousands of particles.
- I/O: read `.cmm` markers and `.em` motive lists; write `.em` and
  STOPGAP `.star`.
- Commands: `placeobject` and `pickparticle`.
- Native ChimeraX session save/restore. GPL v3.

[Unreleased]: https://github.com/baradlab/Geopickr/compare/v1.1.0b3...HEAD
[1.1.0b3]: https://github.com/baradlab/Geopickr/releases/tag/v1.1.0b3
[1.1.0b2]: https://github.com/baradlab/Geopickr/releases/tag/v1.1.0b2
[1.1.0b1]: https://github.com/baradlab/Geopickr/releases/tag/v1.1.0b1
[1.0.0]: https://github.com/baradlab/Geopickr/releases/tag/v1.0.0
