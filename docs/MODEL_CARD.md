# Model card — RouteShield segment risk

**Version** 1.0.0 · MIT

## Intended use

Pre-dispatch route risk scoring and safer-alternative recommendation for
commercial fleets. Decision-support for a human dispatcher.

**Not** for: individual driver discipline or performance management, insurance
pricing, enforcement, or any automated decision affecting a person.

## Inputs

Per road segment: curvature, junction density, street lighting, gradient, speed
limit, lane count, fleet exposure (vehicle-km), and historical telematics
events with severity and timestamp. At request time: departure hour, weather
bucket, optional driver cohort.

## Output

Posterior distribution over the event rate per vehicle-km for each road, giving
a mean, a 90% credible interval and a High/Medium/Low confidence badge; route
risk as the sum over segments, with a Monte Carlo interval.

## Training

Frequency: Gamma-Poisson with a Poisson gradient-boosting prior (blended with a
GLM), dispersion fitted by Negative-Binomial marginal likelihood, posterior
variance inflated by the prior's out-of-fold log error. Severity: per-road mean
severity shrunk toward the network mean with 8 pseudo-events.

## Evaluation

Temporal holdout, 9 months → 4 months. See the README table.

## Limitations

1. **Trained on synthetic events.** No public Indian fleet-telematics dataset
   exists. All metrics validate the pipeline, not real-world accuracy.
2. **Exposure is required.** Without vehicle-km the model cannot distinguish
   busy from risky — the central failure it exists to fix.
3. **Context multipliers are priors, not fitted** in the default configuration.
   `calibrate_from_events()` refits them given hour-of-day exposure.
4. **Gradient is a corridor attribute** in the bundled network; a real DEM is
   required for ghat sections.
5. **Map matching drops ~13% of events** by design, near junctions. Those are
   exactly the locations a black-spot analysis cares about, so hotspot counts
   near complex junctions are conservative.
6. **No causal claim.** The model predicts where incidents occur; it does not
   establish that re-routing causes fewer of them. Shadow mode is the test.
7. **Feedback loop risk.** If routes are chosen to avoid high-risk roads, those
   roads accumulate less exposure and their estimates degrade. Production must
   retain a shadow sample on avoided roads.

## Ethical considerations

Cohort-level reporting only, by construction. Risk scores reflect road
infrastructure, not driver quality; presenting them as driver performance would
be a misuse. Advisory with logged overrides keeps a human accountable.
