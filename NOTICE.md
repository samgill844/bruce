# BRUCE: copyright, licence and attribution

BRUCE is distributed under the GNU General Public License, version 3 or (at your
option) any later version (`GPL-3.0-or-later`). The complete licence is in
[LICENSE](LICENSE).

BRUCE contributions: Copyright (C) 2026 Samuel Gill and contributors.

## Adapted ellc routines

BRUCE's orbital geometry and radial-velocity core is adapted from the Fortran
source of **ellc**, by Pierre Maxted:

- Upstream source: https://github.com/pmaxted/ellc
- Upstream copyright notice: Copyright (C) 2016 Pierre Maxted.
- Upstream licence: GNU GPL version 3 or, at the recipient's option, any later
  version; see the notices in `f_src/ellc.f90` and the upstream `LICENSE.txt`.
- Reference: Maxted (2016), *Astronomy & Astrophysics*, 591, A111,
  https://doi.org/10.1051/0004-6361/201628579.

BRUCE adapts these routines to C, adds NumPy interfaces and OpenMP evaluation,
uses the analytical qpower2 transit approximation, and modifies aspects of the
orbital implementation, including phase wrapping and eclipse-epoch conversion.
The original copyright and authorship of the adapted work remain with its
respective rights holders.

The analytical transit model follows Maxted & Gill (2019), *Astronomy &
Astrophysics*, 622, A33, https://doi.org/10.1051/0004-6361/201834563. The
elliptic Kepler solver follows Markley (1995), *Celestial Mechanics and Dynamical
Astronomy*, 63, 101–111, https://doi.org/10.1007/BF00691917.

## Distribution terms

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

Third-party dependencies and externally obtained astronomical data retain their
own licences and attribution requirements. This notice does not relicense those
materials or claim ownership of upstream work.
