---
title: 'BRUCE: Analytical exoplanet transit modelling and photometric analysis in Python and C'
tags:
  - Python
  - C
  - astronomy
  - exoplanets
  - time-series analysis
authors:
  - name: Samuel Gill
    orcid: 0000-0002-4259-0155
    affiliation: 1
affiliations:
  - name: University of Warwick, United Kingdom
    index: 1
date: 3 November 2026
bibliography: paper.bib
---

# Summary

A planet crossing its host star causes a small decrease in observed brightness. The shape and repetition of these transits constrain its relative size and orbit. Interpreting observations requires physical models, data preparation and parameter estimation. Repeated model evaluations become computationally demanding during fitting and searches of large datasets.

BRUCE is a Python and C package for analysing transiting exoplanets and related stellar systems. Its C modelling core is adapted from the Fortran-based `ellc` binary-star model [@Maxted2016]. It combines analytical transit and radial-velocity models with photometric processing, transit-template searches and parameter-estimation utilities. It also provides tools for retrieving observations from the Transiting Exoplanet Survey Satellite (TESS), investigating ambiguous periods and visualising orbital configurations. Its numerical routines are independently accessible. Source code and documentation are available in the [BRUCE repository](https://github.com/samgill844/bruce).

# Statement of need

TESS [@Ricker2015] has identified thousands of exoplanet candidates and produced extensive archives of light curves and full-frame images [@Guerrero2021]. Searching these data and fitting the resulting events require repeated model evaluations across many stars, epochs and trial parameters. The Next Generation Transit Survey (NGTS) [@Wheatley2018] provides high-cadence survey photometry and extended monitoring of many long-period planet candidates from the TESS mission. Recovery campaigns [@Gill2020TIC; @Gill2020Binary; @Gill2020NGTS11] and subsequent work [@Gill2024] exemplify the large, often discontinuous datasets that must be searched for sparse transit signals. Efficient models are therefore needed both to scan these observations and to fit events jointly with TESS data. PLATO's planned long-baseline observations of large stellar samples [@Rauer2025] will extend the need for computationally efficient detection and characterisation tools.

BRUCE combines fast modelling with data retrieval, normalisation, binning, template searches and period-alias assessment. A fitted transit can become a search template, and the resulting events can constrain follow-up observations. This package provides a toolkit to find a fit exoplanet systems in photometric data and allows users to hand-roll custom fitting routines as required.

# State of the field

Several established packages address parts of this problem. `batman` provides transit calculations for a range of limb-darkening laws [@Kreidberg2015]. `ellc` models detached eclipsing binaries and transiting planets with a broader treatment of stellar geometry and binary effects [@Maxted2016]. PyTransit provides efficient transit models and interfaces designed for demanding photometric analyses [@Parviainen2015], including a documented [power-2 implementation](https://pytransit.readthedocs.io/en/stable/models/qpower2.html). Lightkurve provides an extensive environment for accessing and analysing Kepler and TESS observations [@Lightkurve2018].

BRUCE builds on the foundations. Its contribution is the combination of a compact C implementation, optional shared-memory parallelism, direct model-and-likelihood evaluation and workflows for matching known transit shapes and investigating period ambiguity. The package focuses on speed and provides a single environment in which exoplanet researchers can find transit signals, fit planetary systems and interpret results.

# Software design

The binary-star modelling core is adapted from `ellc` [@Maxted2016], with an analytical eccentric-anomaly solver based on @Markley1995: a cubic approximation followed by a fifth-order correction solves the elliptic Kepler equation. The resulting true anomaly determines Keplerian radial velocities and the projected separation used for photometric transits. Transit evaluation focuses on the fast analytical `qpower2` approximation [@MaxtedGill2019], using the power-2 limb-darkening law. Avoiding repeated numerical integration over the stellar disc makes this approximation efficient for fitting exoplanet transits but comes at the cost of accuracy as the radius ratio increases (see [@MaxtedGill2019]). Figure \ref{fig:transit} compares the analytical transit model with independent numerical integration for a typical exoplanet system.

Python supplies array-based interfaces, data handling and fitting orchestration, while C with OpenMP handles repeated calculations. NumPy arrays connect these layers through the [Numpy C API][https://numpy.org/doc/stable/reference/c-api/index.html]. OpenMP distributes sample-wise calculations across CPU threads, further accelerating fitting and minimisation routines. A combined evaluation mode accumulates a likelihood statistic without returning the full model array, although its performance advantage depends on workload and memory costs. Numerical cadence integration requires odd subsample counts of at least three, with a default of 11, centred across each exposure. Keplerian radial velocities support primary and double-lined systems, and sampling helpers interface with `emcee` [@ForemanMackey2013] and SciPy-based minimisation routines [@Virtanen2020].

The higher-level modules support TESS data retrieval, photometric time-series preparation, transit fitting, template matching and period-alias assessment. Detrending remains an explicit modelling choice: this package provides filter routines that can remove stellar variability but also distort a transit if not used appropriately. Plotting, sampler/minimizer helpers and orbit animations support interpretation of fitted systems.

## Use cases

**Direct transit fitting.** BRUCE provides fitting routines and sampling utilities that simplify preparing photometric data, fitting transit models and interpreting the results. Supplying photometry and uncertainties to `bruce.binarystar.lc` evaluates the model and accumulates the likelihood statistic in C, avoiding transfer of a model array at every optimiser call or sampler step. The current wrapper returns negative log likelihood, so posterior sampling requires the corresponding sign conversion.

**Detecting single transits.** Transit templates can be compared to photometric data over a range of trial epochs to search for transit events. Published examples of this method include NGTS-11 b [@Gill2020NGTS11] and TOI-2447 b/NGTS-29 b [@Gill2024]. The latter uses a TESS-derived template to recover events in archival NGTS survey observations and subsequent monitoring. These methods apply to both ground-based and space-based photometry, including searches through TESS data for additional transits of long-period planetary systems. Such searches can be computationally expensive, but BRUCE provides fast template matching capabilities. Detection thresholds must be calibrated for each dataset and search.

**Resolving duotransits.** Two separated transits permit discrete aliases. BRUCE provides a period-resolution framework that organises transit events, generates candidate aliases and combines event fitting, observing coverage and searches for additional events to assess the possible periods. This framework simplifies the analysis of duotransit systems such as those identified in TESS samples [@Hawthorn2024].

## Model comparison

BRUCE is compared with `ellc` numerical integration using identical power-2 coefficients [Figure \ref{fig:transit}]. No parameters are fitted. For this case, the maximum difference is 74.0 ppm; changing `ellc` from fine to very-fine grids changes the reference by at most 0.58 ppm. Source code and model arrays accompany the paper.

![Analytical versus numerical power-2 transit models for spherical bodies with scaled stellar radius 0.05, radius ratio 0.1, zero impact parameter and eccentricity, and instantaneous sampling. Both use c=0.715452 and alpha=0.643777. Upper: BRUCE and very-fine ellc integration. Lower: differences from that reference, including sparse ellc integration for context. This matched-law comparison follows the layout of Maxted & Gill (2019), Figure 3. \label{fig:transit}](figures/ellc_comparison.png){width="95%"}

## Computational performance

BRUCE is a fast light-curve framework and its speed is demonstrated with a benchmark test that records model-only calls and direct C likelihood calls with `flux` and `flux_err` supplied to `bruce.binarystar.lc`. Figure \ref{fig:performance} shows the latter, the relevant repeated evaluation for fitting. Contiguous double-precision arrays contain 100 to 100,000 samples; seven batches follow warm-up in separate one- and four-thread processes. The instantaneous circular case uses $e=0$, the instantaneous eccentric case $e=0.4$, and the exposure-integrated case $e=0$ with 11 subsamples over 2 minutes. All use fixed uncertainties. Imports, input construction, I/O and optimiser execution are excluded; settings, raw timings, source hashes and the compiler log are retained.

On an Intel Core i7-9750H at a nominal 2.60 GHz, with six physical cores, Python 3.9.13, NumPy 1.26.4 and Apple Clang 21.0.0 with OpenMP, the 1,000-sample circular likelihood achieved approximately 7,500 evaluations per second with one thread and 28,000 with four. For 100,000 samples, the rates were 57 and 280 evaluations per second, a speed-up of 4.90. At that size, the one-thread eccentric and exposure-integrated rates were 25.6 and 11.0 evaluations per second. Rates describe complete model-and-likelihood evaluations, not independent posterior samples.

![Direct C likelihood throughput and OpenMP scaling for complete model evaluations. Left: reciprocal median call time with one thread; shading spans reciprocals of the upper and lower timing quartiles. Right: one-thread divided by four-thread median call time. Labels specify eccentricity and exposure treatment. The same retained benchmark data underpin both panels; these are local BRUCE measurements, not comparisons with competing packages.\label{fig:performance}](figures/performance.png){width="95%"}

These tests are hardware dependent; multiprocessing analyses should control OpenMP threads to avoid oversubscription. Users should be aware of the `OMP_NUM_THREADS` environment variable to control the number of threads used when calculating a log-likelihood or generating a model. While OpenMP is resilient, users should set it to 1 when fitting data with multiprocessing (e.g. with emcee).

# Research impact statement

BRUCE grew from the need to search and fit large photometric datasets efficiently, particularly TESS and NGTS long-period candidates. The author's recovery programme includes TIC-238855958, TIC-231005575 and NGTS-11 b [@Gill2020TIC; @Gill2020Binary; @Gill2020NGTS11], followed by TOI-2447 b/NGTS-29 b [@Gill2024]; the TESS duotransit catalogue of @Hawthorn2024 further motivates tools for ambiguous periods and targeted monitoring. These studies motivate BRUCE's workflows; they do not all use the present release.

BRUCE provides undergraduate and postgraduate students, as well as established researchers, with tools to find, fit and prepare planetary-system analyses for publication. The educational and research scope of BRUCE lowers the amount of processing needed to move between candidate detection, parameter estimation and follow-up planning. Reproducible tests, benchmarks and regression checks are provided.

# AI usage disclosure

OpenAI Codex assisted with manuscript drafting, bibliography preparation, benchmark and figure scripts, documentation, and preceding numerical refactoring and regression-test development. The author subsequently edited the text substantially and checked that the text and figures reflect the intended content. The author used Codex assistance for the NumPy C API bindings. Automated checks included compilation and regression tests.

# Acknowledgements

Samuel Gill acknowledges support from the Science and Technology Facilities Council (STFC) through grant ST/X001121/1. The author thanks Pierre Maxted for developing and sharing `ellc` [@Maxted2016], from which BRUCE's modelling core was adapted. BRUCE uses Astropy, a community-developed astronomy package [@Astropy2022], and NumPy [@Harris2020]. The author also acknowledges the Mikulski Archive for Space Telescopes (MAST) at the Space Telescope Science Institute for archiving and providing access to TESS observations used in BRUCE workflows.

# References
