# Author review before JOSS submission

The present manuscript is a first draft, not a claim of completed journal readiness. Samuel Gill, University of Warwick and ORCID 0000-0002-4259-0155 were supplied by the author. No co-authors, external users or archive DOI have been inferred. STFC grant ST/X001121/1 was supplied by the author and is now acknowledged, together with ellc, Astropy, NumPy and MAST. The revised research context cites the Gill (2020, three papers; 2024) and Hawthorn (2024) studies requested by the author, with publication details verified.

## Scientific content to complete

1. Confirm which cited Gill/Hawthorn studies used BRUCE directly or used precursor code. The revised **Research impact statement** identifies the programme that motivated the software without asserting that every 2020 analysis used the current package. Add specific student-led applications if available; the manuscript describes student use as a capability, not as a measured adoption claim.
2. Consider replacing the synthetic fit with a real TESS transit and a ground-based event or period-alias figure. Supply programme/target identifiers, data access details and attribution. The present synthetic fit fixes period, inclination and limb darkening; it does not report credible intervals or demonstrate full parameter recovery.
3. Review the explanation of why BRUCE's connected event/search workflow justified building a package alongside existing transit engines. This is drafted from the current code and should reflect the author's actual design rationale.
4. Independently validate the implemented flux approximation and exposure integration against reference calculations over the scientific domain to be claimed. The benchmark and orbital tests do not establish this physical validation. The qpower2 publication's performance or accuracy cannot be transferred automatically to this implementation.
5. Check the supplied STFC acknowledgement and add any further funding, institutional support or authors. The software/archive acknowledgements now explicitly name ellc, Astropy, NumPy and MAST.

## Verified references and scope

The eccentric-anomaly expressions match Markley (1995), DOI 10.1007/BF00691917: an analytical cubic starter plus a fifth-order correction. The public `lc` evaluator currently calls only the power-2 law (`ld_law=2` or its single-event variant `-2`). Other profiles and annulus-integration routines exist in C, but their calls in `__lc` are commented out; the paper does not claim that they are active Python options.

Gill (2024), DOI 10.1093/mnras/stae1256, is the TOI-2447 b/NGTS-29 b study. It describes searching archival NGTS **survey** data and follow-up monitoring using a TESS-derived template. That is an unknown-epoch search with a known shape, rather than an entirely untargeted shape-and-period survey. The use-case paragraph separately describes how a template bank can support a blind search. The synthetic figure is not a reproduction of the published data or a claim of survey-wide completeness.

The three Gill (2020) entries are TIC-238855958 (10.1093/mnras/stz3212), TIC-231005575 (10.1093/mnras/staa1248) and NGTS-11 b (10.3847/2041-8213/ab9eb9). Local preview citations disambiguate their years; the official JOSS cite processor determines its own author/year formatting. The two binary studies are described as candidate-recovery work, not confirmed planets. PLATO is cited using the published 2025 Rauer mission paper rather than its 2024 preprint.

## Performance and a defect exposed during drafting

The raw benchmark includes one- and four-thread timings for model-only, fused negative-log-likelihood and model-plus-NumPy operations. The combined path is not uniformly faster: the paper deliberately states this rather than claiming a general advantage. The numbers are local microbenchmarks, not end-to-end fit timings or comparisons with other packages.

An initial equivalence check with `offset=1` failed: the fused result omitted the expected variance-dependent term. In `src/bruce_c/bruce_numpy.c`, `lc_loglike` declares `offset` as a `double`, but its `PyArg_ParseTuple` format ends in `i`, which requires an `int *`. This is a C argument-type mismatch and should be repaired and regression-tested before claiming support for that likelihood mode. **No package source was changed as part of manuscript preparation.** The retained timings use `offset=0`, fixed uncertainties and zero jitter, and passed direct equivalence checks. Because the parse mismatch is still present, repair it and regenerate these measurements before submission rather than treating this setting as a complete fix.

The draft measurements identify the CPU as an Intel Core i7-9750H, six physical/twelve logical cores, with nominal 2.60 GHz frequency. Frequency was not fixed or monitored. Do not infer energy use, maximum sustained throughput or scalability beyond the measured four threads. Repeat on an otherwise idle machine if using these results in the final submission. Save the final commit and release information alongside the raw data.

## Submission materials

- Confirm an OSI-approved software licence and add a root licence file before submission. No tracked licence file was found during this setup. Choosing a licence requires the copyright holder's decision; this draft does not choose one.
- Check installation from a clean environment, public issue/contribution/support guidance and the complete functionality claimed in the article. These are part of JOSS software review rather than manuscript typography alone.
- Create a versioned release and appropriate archival record at the stage requested by the JOSS editor; add the final software archive citation/DOI. Do not invent a DOI or imply a release has been archived.
- Run the official Inara build and inspect its PDF. Only the local reading preview was rendered here because Docker, Pandoc and `latexmk` were unavailable.
- Complete the **AI usage disclosure** with the exact tool/model version and an accurate statement of the human review actually performed. The draft documents assistance with manuscript, scripts, documentation, refactoring and tests; it deliberately does not assert that the author has already reviewed or validated everything. Do not submit its future-tense review instructions unchanged.
- Remove the `AUTHOR INPUT` comments after completing them. They do not appear in the reading PDF.

Current guidance consulted on 3 November 2026: [paper format](https://joss.readthedocs.io/en/latest/paper.html), [submission requirements and AI policy](https://joss.readthedocs.io/en/latest/submitting.html), and [review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html).
