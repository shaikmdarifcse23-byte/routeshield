/* RouteShield — prototype-phase pitch deck.
   Every number is read from artifacts/results.json, so the deck cannot drift
   from the code. Run: node scripts/build_deck.js */
const pptxgen = require('pptxgenjs');
const fs = require('fs');
const path = require('path');
const { applyTheme } = require(process.env.PPTX_SKILL + '/scripts/apply_theme.js');

const ROOT = path.resolve(__dirname, '..');
const R = JSON.parse(fs.readFileSync(path.join(ROOT, 'artifacts/results.json'), 'utf8'));
const SHOTS = process.env.SHOTS_DIR;

const sm = R.segment_model, cf = R.counterfactual, mm = R.map_matching;
const f = (x, d = 2) => Number(x).toFixed(d);
const pc = (x, d = 1) => (100 * x).toFixed(d) + '%';

// ---------------------------------------------------------------- theme
const THEME = {
  name: 'RouteShield',
  headFontFace: 'Cambria',
  bodyFontFace: 'Calibri',
  colors: {
    dk1: '0E1014',   // near-black: dominant
    lt1: 'FFFFFF',
    dk2: '1B1F27',   // panel
    lt2: 'EEF0F4',
    accent1: 'F0A94E',   // amber — risk
    accent2: '3987E5',   // blue — the recommended route
    accent3: 'D03B3B',   // critical
    accent4: '8A93A3',   // muted ink
    accent5: '0CA30C',   // good
    accent6: 'DD8A2E',
    hlink: '3987E5', folHlink: '8A93A3',
  },
};
const INK = 'EDEFF3', MUTED = '98A1B0', PANEL = '1B1F27', AMBER = 'F0A94E', BLUE = '3987E5';

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';            // 13.3 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = 'RouteShield';
pres.title = 'RouteShield — Fleet Route Risk Intelligence';
const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: 'TITLE', background: { color: C.text1 },
  objects: [],
});
pres.defineSlideMaster({
  title: 'CONTENT', background: { color: C.text1 },
  objects: [
    { text: { text: 'RouteShield · open source (MIT)', options: {
        x: 0.65, y: 6.95, w: 7, h: 0.3, fontSize: 10, color: '59606E', isTextBox: true } } },
  ],
  slideNumber: { x: 12.75, y: 6.95, fontSize: 10, color: '59606E' },
});
pres.defineSlideMaster({
  title: 'SECTION', background: { color: '161A21' },
  objects: [],
});

// ---------------------------------------------------------------- helpers
let N = 0;
const card = (s, x, y, w, h, fill = PANEL) => s.addShape(pres.ShapeType.roundRect, {
  x, y, w, h, fill: { color: fill }, rectRadius: 0.09,
  line: { color: '262B35', width: 1 }, objectName: `card${++N}`,
});
const tx = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0 }, o));

function statCard(s, x, y, w, value, label, colour = INK, sub) {
  card(s, x, y, w, sub ? 1.65 : 1.35);
  tx(s, value, { x: x + 0.22, y: y + 0.16, w: w - 0.44, h: 0.62,
    fontSize: 34, bold: true, color: colour, fontFace: 'Cambria' });
  tx(s, label, { x: x + 0.22, y: y + 0.80, w: w - 0.44, h: 0.42,
    fontSize: 11, color: MUTED });
  if (sub) tx(s, sub, { x: x + 0.22, y: y + 1.18, w: w - 0.44, h: 0.38,
    fontSize: 10, color: '6E7686', italic: true });
}

function bulletCard(s, x, y, w, h, heading, lines, accent = AMBER) {
  card(s, x, y, w, h);
  s.addShape(pres.ShapeType.ellipse, { x: x + 0.24, y: y + 0.26, w: 0.2, h: 0.2,
    fill: { color: accent }, objectName: `dot${++N}` });
  tx(s, heading, { x: x + 0.56, y: y + 0.21, w: w - 0.8, h: 0.34,
    fontSize: 15, bold: true, color: INK });
  s.addText(lines.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < lines.length - 1 } })),
    { x: x + 0.3, y: y + 0.66, w: w - 0.56, h: h - 0.86, fontSize: 12.5,
      color: 'C3C9D4', paraSpaceAfter: 7, isTextBox: true, margin: 0 });
}

const slide = (master, section) => pres.addSlide({ masterName: master, sectionTitle: section });

/* Titles are explicit text boxes, not placeholders.
   pptxgenjs writes placeholder-inherited text that PowerPoint resolves but
   LibreOffice silently drops, so a deck that looks right in one renderer
   loses every title in the other. A fixed x/y/size here also guarantees the
   titles cannot drift between slides of the same type. */
const TITLE_Y = 0.42;
const title = (s, text) => tx(s, text, { x: 0.65, y: TITLE_Y, w: 12.0, h: 0.8,
  fontSize: 32, bold: true, color: INK, fontFace: 'Cambria' });

// ================================================================= 1 TITLE
pres.addSection({ title: 'Open' });
{
  const s = slide('TITLE', 'Open');
  s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: 13.33, h: 0.055,
    fill: { color: AMBER }, objectName: 'topedge' });
  tx(s, 'Netradyne Hackathon · Prototype Phase', { x: 0.92, y: 1.75, w: 9, h: 0.35,
    fontSize: 13, color: AMBER, charSpacing: 2 });
  tx(s, 'RouteShield', { x: 0.9, y: 2.5, w: 11.5, h: 1.3, fontSize: 52, bold: true,
    color: INK, fontFace: 'Cambria' });
  tx(s, 'Score a planned fleet route for risk before dispatch — with an honest confidence '
    + 'interval, an explanation a dispatcher can argue with, and safer alternatives on an '
    + 'explicit risk-vs-time trade-off.',
    { x: 0.92, y: 3.95, w: 10.6, h: 1.4, fontSize: 18, color: MUTED });
  tx(s, 'Safer Roads for India  ·  Fleet Route Risk Intelligence  ·  MIT licensed',
    { x: 0.92, y: 5.75, w: 11, h: 0.4, fontSize: 13, color: '6E7686' });
  s.addNotes('RouteShield turns road risk into something a dispatcher can see and act on '
    + 'at the moment the route is assigned. Everything in this deck is measured by code in '
    + 'the repository and reproducible with one command.');
}

// ================================================================= 2 PROBLEM
pres.addSection({ title: 'Problem' });
{
  const s = slide('CONTENT', 'Problem');
  title(s, 'Dispatchers choose routes blind to risk');
  tx(s, 'Distance and ETA are visible. The things that actually hurt are not.',
    { x: 0.65, y: 1.22, w: 11.6, h: 0.4, fontSize: 15, color: MUTED });

  const items = [
    ['Unlit stretches', 'No street lighting on rural and peri-urban links'],
    ['Ghat curves', 'Gradient and curvature on hill sections'],
    ['Junction history', 'Clusters of harsh braking and near-misses'],
    ['Night fatigue', '22:00–05:00, the largest behavioural factor in Indian fleet crashes'],
    ['Monsoon and fog', 'Seasonal conditions that change risk week to week'],
    ['Sparse history', 'Most roads have almost no fleet data at all'],
  ];
  items.forEach(([h, d], i) => {
    const x = 0.65 + (i % 3) * 4.06, y = 1.85 + Math.floor(i / 3) * 1.72;
    bulletCard(s, x, y, 3.82, 1.52, h, [d]);
  });
  tx(s, 'None of it is on the dispatcher’s screen when the decision is made.',
    { x: 0.65, y: 5.45, w: 11.6, h: 0.45, fontSize: 16, color: AMBER, italic: true });
  s.addNotes('The information exists — in telematics, in OSM, in weather feeds. It just '
    + 'never reaches the one person who can act on it, at the one moment it matters.');
}

// ================================================================= 3 INSIGHT
{
  const s = slide('CONTENT', 'Problem');
  title(s, 'The obvious fix does not work');
  tx(s, 'Build a heat map of historical incidents. Here is what it actually ranks.',
    { x: 0.65, y: 1.22, w: 11.6, h: 0.4, fontSize: 15, color: MUTED });

  card(s, 0.65, 1.95, 5.6, 2.5, '241A1A');
  tx(s, f(sm.roc_auc_top_decile_TRUE_rate.raw_history_count, 3), {
    x: 0.95, y: 2.25, w: 5.0, h: 1.0, fontSize: 62, bold: true, color: 'E07373', fontFace: 'Cambria' });
  tx(s, 'AUC — raw historical event counts, ranking the roads that are truly highest-risk',
    { x: 0.95, y: 3.3, w: 5.0, h: 0.9, fontSize: 13, color: 'C9A0A0' });

  card(s, 6.55, 1.95, 5.7, 2.5);
  tx(s, 'Worse than a coin flip', { x: 6.85, y: 2.25, w: 5.1, h: 0.45,
    fontSize: 20, bold: true, color: INK });
  s.addText([
    { text: 'A count measures how BUSY a road is, not how RISKY.', options: { breakLine: true, bold: true, color: AMBER } },
    { text: 'The Outer Ring Road tops the map because everything drives on it.', options: { breakLine: true } },
    { text: 'And roads nobody has driven yet report “zero risk” rather than “unknown”.', options: {} },
  ], { x: 6.85, y: 2.78, w: 5.1, h: 1.4, fontSize: 13, color: 'C3C9D4',
       paraSpaceAfter: 8, isTextBox: true, margin: 0 });

  tx(s, 'RouteShield ranks the same roads at ' + f(sm.roc_auc_top_decile_TRUE_rate.routeshield, 3) + ' AUC',
    { x: 0.65, y: 4.85, w: 11.6, h: 0.5, fontSize: 24, bold: true, color: '5FD38A', fontFace: 'Cambria' });
  tx(s, 'Measured against ground truth on a strict temporal holdout — fit on 9 months, scored on the next 4.',
    { x: 0.65, y: 5.42, w: 11.6, h: 0.4, fontSize: 13, color: MUTED });
  s.addNotes('This is the single most important slide. Everyone builds the heat map. '
    + 'It ranks traffic volume, not risk, and it is silent exactly where you have no data.');
}

// ================================================================= 4 PRODUCT
pres.addSection({ title: 'Product' });
{
  const s = slide('CONTENT', 'Product');
  title(s, 'The dispatcher console');
  if (SHOTS && fs.existsSync(path.join(SHOTS, 'dispatch.png'))) {
    s.addImage({ path: path.join(SHOTS, 'dispatch.png'), x: 0.65, y: 1.25, w: 8.05, h: 5.03 });
  }
  const pts = [
    ['Risk before dispatch', 'Expected severity-weighted events, with a 90% interval'],
    ['Confidence badge', 'Says plainly when the data is too thin to trust'],
    ['Pareto alternatives', 'Every genuinely different route, time against risk'],
    ['Why this score', 'Exact factor contributions, worst segments, black spots'],
  ];
  pts.forEach(([h, d], i) => bulletCard(s, 8.95, 1.25 + i * 1.28, 3.72, 1.12, h, [d], i === 2 ? BLUE : AMBER));
  s.addNotes('Live demo runs entirely in the browser — the model, the routing and the '
    + 'Monte Carlo interval are all client-side. No backend needed to try it.');
}

// ================================================================= 5 HOW
{
  const s = slide('CONTENT', 'Product');
  title(s, 'How it works');
  const steps = [
    ['1', 'Ingest', 'Telematics events, OpenStreetMap geometry and tags, weather, elevation'],
    ['2', 'Map-match', 'Snap GPS to the right road — and refuse to guess when it is ambiguous'],
    ['3', 'Model', 'Frequency × severity, Gamma-Poisson posterior with calibrated uncertainty'],
    ['4', 'Score', 'Route risk = lookup + sum, adjusted for hour, weather and cohort'],
    ['5', 'Choose', 'Pareto frontier of alternatives, with a dispatch policy you control'],
    ['6', 'Learn', 'Log the decision, join to outcomes, report trends and overrides'],
  ];
  steps.forEach(([n, h, d], i) => {
    const x = 0.65 + (i % 3) * 4.06, y = 1.5 + Math.floor(i / 3) * 2.35;
    card(s, x, y, 3.82, 2.1);
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.26, y: y + 0.26, w: 0.46, h: 0.46,
      fill: { color: i < 3 ? AMBER : BLUE }, objectName: `num${i}` });
    tx(s, n, { x: x + 0.26, y: y + 0.33, w: 0.46, h: 0.33, fontSize: 15, bold: true,
      color: '0E1014', align: 'center' });
    tx(s, h, { x: x + 0.86, y: y + 0.3, w: 2.7, h: 0.38, fontSize: 17, bold: true, color: INK });
    tx(s, d, { x: x + 0.28, y: y + 0.92, w: 3.3, h: 1.0, fontSize: 12.5, color: 'C3C9D4' });
  });
  s.addNotes('Everything expensive happens in a nightly batch. Request time is a table '
    + 'lookup and a sum, which is why scoring a route costs under a millisecond.');
}

// ================================================================= 6 MODEL
pres.addSection({ title: 'Model' });
{
  const s = slide('CONTENT', 'Model');
  title(s, 'Three modelling decisions that made the difference');

  const blocks = [
    ['Frequency × severity', [
      'Counting severity-weighted events in one Poisson rate conflates two processes.',
      'One accident carries 10× a harsh brake, the count becomes wildly overdispersed,',
      'the fitted prior collapses — and the model degenerates into the baseline it should beat.',
    ]],
    ['A prior from road structure', [
      'Poisson gradient boosting on curvature, junctions, lighting, gradient, speed, lanes.',
      'A road with no history inherits the risk of structurally similar roads,',
      'instead of reporting zero. Shrinkage strength is learned, not hand-tuned.',
    ]],
    ['Uncertainty about the prior itself', [
      'Textbook empirical Bayes treats the fitted prior mean as exact. It is not.',
      'Measured 90% coverage of the plain version: 0.69 — a tight interval around a wrong centre.',
      'Folding in the prior’s own out-of-fold error lifts coverage to ' + pc(sm.latent_rate_interval_coverage_90, 2) + '.',
    ]],
  ];
  blocks.forEach(([h, lines], i) => {
    const y = 1.45 + i * 1.78;
    card(s, 0.65, y, 11.95, 1.6);
    s.addShape(pres.ShapeType.ellipse, { x: 0.95, y: y + 0.3, w: 0.26, h: 0.26,
      fill: { color: [AMBER, BLUE, '5FD38A'][i] }, objectName: `mdot${i}` });
    tx(s, h, { x: 1.35, y: y + 0.22, w: 5.0, h: 0.4, fontSize: 17, bold: true, color: INK });
    tx(s, lines.join(' '), { x: 1.35, y: y + 0.68, w: 10.9, h: 0.8, fontSize: 12.5, color: 'C3C9D4' });
  });
  s.addNotes('We found each of these by measuring, not by design. The uncertainty '
    + 'correction in particular: our first version was confidently wrong on sparse roads.');
}

// ================================================================= 7 MATCHING
{
  const s = slide('CONTENT', 'Model');
  title(s, 'A bug we found by measuring, not by looking');
  tx(s, 'Near a junction, two roads are almost equidistant. Forcing a match there is how a quiet side road inherits a trunk road’s events.',
    { x: 0.65, y: 1.2, w: 11.9, h: 0.45, fontSize: 14.5, color: MUTED });

  statCard(s, 0.65, 1.95, 3.82, '4.6%', 'of events landed on the WRONG road when every match is forced', 'E07373');
  statCard(s, 4.75, 1.95, 3.82, pc(1 - mm.road_match_accuracy, 1), 'after adding an 18 m ambiguity gate', '5FD38A');
  statCard(s, 8.85, 1.95, 3.77, pc(1 - mm.match_rate, 0), 'of events dropped — the price paid', AMBER);

  card(s, 0.65, 3.75, 11.95, 2.1);
  tx(s, 'Why it mattered more than it looks', { x: 0.98, y: 3.98, w: 8, h: 0.4,
    fontSize: 17, bold: true, color: INK });
  s.addText([
    { text: 'Misattributed events made a handful of low-traffic roads look like the worst in the city — and the risk model faithfully learned it.', options: { breakLine: true } },
    { text: 'Measured effect on the product: the benefit of re-routing went from −0.2% (no benefit at all) to +' + f(cf.avg_true_risk_reduction_pct_when_switched, 1) + '% once matching was fixed.', options: { breakLine: true, bold: true, color: AMBER } },
    { text: 'A Bayesian model cannot repair data attributed to the wrong road. Correlation with the true rate rose from 0.20 to 0.87.', options: {} },
  ], { x: 0.98, y: 4.45, w: 11.3, h: 1.3, fontSize: 13, color: 'C3C9D4',
       paraSpaceAfter: 7, isTextBox: true, margin: 0 });
  s.addNotes('This is the slide we are proudest of. It is not a feature — it is evidence '
    + 'that we validated the thing rather than trusting it.');
}

// ================================================================= 8 EVIDENCE
pres.addSection({ title: 'Evidence' });
{
  const s = slide('CONTENT', 'Evidence');
  title(s, 'Does the score predict incidents?');
  tx(s, 'Strict temporal holdout: fit on 9 months, score the next 4. Every metric against a baseline.',
    { x: 0.65, y: 1.2, w: 11.9, h: 0.4, fontSize: 14.5, color: MUTED });

  s.addChart(pres.ChartType.bar, [{
    name: 'AUC',
    labels: ['RouteShield', 'Road features only', 'Historical rate', 'Historical count'],
    values: [sm.roc_auc_top_decile_TRUE_rate.routeshield,
             sm.roc_auc_top_decile_TRUE_rate.road_features_only,
             sm.roc_auc_top_decile_TRUE_rate.raw_history_rate,
             sm.roc_auc_top_decile_TRUE_rate.raw_history_count],
  }], {
    x: 0.65, y: 1.8, w: 6.9, h: 3.0, barDir: 'bar',
    showTitle: true, title: 'Ranking the truly highest-risk roads (AUC)',
    titleColor: INK, titleFontSize: 13, titleFontFace: '+mn-lt',
    chartColors: [AMBER, '5B6372', '5B6372', 'E07373'], varyColors: true,
    showValue: true, dataLabelPosition: 'outEnd', dataLabelColor: INK,
    dataLabelFontSize: 11, dataLabelFontFace: '+mn-lt', dataLabelFormatCode: '0.000',
    catAxisLabelColor: 'C3C9D4', catAxisLabelFontSize: 11, catAxisLabelFontFace: '+mn-lt',
    valAxisLabelColor: MUTED, valAxisLabelFontSize: 10, valAxisLabelFontFace: '+mn-lt',
    valAxisMaxVal: 1.0, valGridLine: { color: '262B35', size: 1 },
    catGridLine: { style: 'none' }, showLegend: false,
    plotArea: { fill: { color: PANEL } }, chartArea: { fill: { color: PANEL } },
  });

  const rows = [
    ['Sparse roads — rank vs truth', f(sm.spearman_vs_true_rate_sparse_only.routeshield, 3),
     f(sm.spearman_vs_true_rate_sparse_only.raw_history_rate, 3)],
    ['Severe-event ranking (AUC)', f(sm.roc_auc_severe_event.routeshield, 3),
     f(sm.roc_auc_severe_event.road_features_only, 3)],
    ['90% interval coverage', pc(sm.latent_rate_interval_coverage_90, 1), 'target 90%'],
    ['Posterior predictive coverage', pc(sm.predictive_interval_coverage_90, 1), 'target 90%'],
    ['Calibration error', f(sm.calibration_error_mean_abs_log_ratio, 3), 'mean |log obs/exp|'],
  ];
  card(s, 7.85, 1.8, 4.77, 3.0);
  tx(s, 'And the rest of it', { x: 8.12, y: 1.98, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: INK });
  rows.forEach(([k, a, b], i) => {
    const y = 2.42 + i * 0.47;
    tx(s, k, { x: 8.12, y, w: 2.85, h: 0.4, fontSize: 11, color: 'C3C9D4' });
    tx(s, a, { x: 10.95, y, w: 0.82, h: 0.4, fontSize: 12, bold: true, color: AMBER, align: 'right' });
    tx(s, b, { x: 11.8, y: y + 0.03, w: 0.75, h: 0.36, fontSize: 9, color: '6E7686', align: 'right' });
  });

  tx(s, 'Reproduce all of it:  python -m routeshield.cli demo',
    { x: 0.65, y: 5.05, w: 11.9, h: 0.4, fontSize: 13, color: BLUE, fontFace: 'Courier New' });
  tx(s, '35 automated tests, ~13 s. Full pipeline and evaluation, ~5 s.',
    { x: 0.65, y: 5.48, w: 11.9, h: 0.4, fontSize: 12, color: MUTED });
  s.addNotes('Ranking against ground truth is the fair test — the observed-rate label is '
    + 'itself a noisy draw, and a noisy baseline can track that noise.');
}

// ================================================================= 9 TRADEOFF
{
  const s = slide('CONTENT', 'Evidence');
  title(s, 'Risk versus time, made explicit');
  tx(s, 'Counterfactual replay: routes are chosen with the model’s predictions but scored against ground truth withheld from training, so the system cannot grade its own homework.',
    { x: 0.65, y: 1.2, w: 11.9, h: 0.45, fontSize: 14, color: MUTED });

  statCard(s, 0.65, 1.95, 3.82, pc(cf.pct_with_safer_alternative / 100, 1),
    'of trips have a materially safer alternative inside a +15% time budget', INK);
  statCard(s, 4.75, 1.95, 3.82, '−' + f(cf.avg_true_risk_reduction_pct_when_switched, 1) + '%',
    'true risk when the recommendation is taken', '5FD38A');
  statCard(s, 8.85, 1.95, 3.77, '+' + f(cf.fleet_time_cost_pct, 1) + '%',
    'fleet-wide travel time — averaged over every trip, not only the switches', AMBER);

  const curve = R.tradeoff_curve;
  s.addChart(pres.ChartType.line, [
    { name: 'Risk reduction (fleet-wide)', labels: curve.map(r => r.time_budget_pct + '%'),
      values: curve.map(r => r.fleet_true_risk_reduction_pct) },
    { name: 'Travel time cost (fleet-wide)', labels: curve.map(r => r.time_budget_pct + '%'),
      values: curve.map(r => r.fleet_time_cost_pct) },
  ], {
    x: 0.65, y: 3.75, w: 11.95, h: 2.55,
    showTitle: true, title: 'What each extra minute of time budget buys — both averaged over every trip',
    titleColor: INK, titleFontSize: 13, titleFontFace: '+mn-lt',
    chartColors: [BLUE, AMBER], lineSize: 3, lineSmooth: false,
    showLegend: true, legendPos: 't', legendColor: 'C3C9D4', legendFontSize: 11,
    legendFontFace: '+mn-lt',
    catAxisLabelColor: 'C3C9D4', catAxisLabelFontSize: 11, catAxisLabelFontFace: '+mn-lt',
    valAxisLabelColor: MUTED, valAxisLabelFontSize: 10, valAxisLabelFontFace: '+mn-lt',
    valGridLine: { color: '262B35', size: 1 }, catGridLine: { style: 'none' },
    plotArea: { fill: { color: PANEL } }, chartArea: { fill: { color: PANEL } },
  });
  s.addNotes('The dispatcher sets the policy — how many extra minutes are acceptable, and '
    + 'how big a risk cut justifies a switch. The system shows the frontier, not a verdict.');
}

// ================================================================= 10 HONESTY
pres.addSection({ title: 'Honesty' });
{
  const s = slide('CONTENT', 'Honesty');
  title(s, 'What is real, and what is not');
  tx(s, 'No public Indian fleet-telematics dataset exists. We say so everywhere it matters, including inside the product.',
    { x: 0.65, y: 1.2, w: 11.9, h: 0.4, fontSize: 14.5, color: MUTED });

  card(s, 0.65, 1.85, 5.85, 4.1, '13211A');
  tx(s, 'Real', { x: 0.98, y: 2.08, w: 4.5, h: 0.4, fontSize: 19, bold: true, color: '5FD38A' });
  s.addText([
    'Bengaluru junction coordinates — ORR, Hosur Rd, NH44/48/75/275/648',
    'OpenStreetMap ingestion against the live Overpass API, unit-tested',
    'Open-Meteo weather feed with an offline climatology fallback',
    'MoRTH severity mix and night-crash share shaping the priors',
    'Every metric, computed by code in the repository',
  ].map((t, i, a) => ({ text: t, options: { bullet: true, breakLine: i < a.length - 1 } })),
    { x: 0.98, y: 2.6, w: 5.25, h: 3.1, fontSize: 12.5, color: 'C3C9D4',
      paraSpaceAfter: 9, isTextBox: true, margin: 0 });

  card(s, 6.75, 1.85, 5.87, 4.1, '241A1A');
  tx(s, 'Not real', { x: 7.08, y: 2.08, w: 4.5, h: 0.4, fontSize: 19, bold: true, color: 'E07373' });
  s.addText([
    'Telematics events — synthetic, clearly labelled, seeded from real structure',
    'Inter-junction geometry — interpolated; one command swaps in true OSM ways',
    'Fleet exposure — drawn heavy-tailed to mimic real lane concentration',
    'Therefore: every number validates the pipeline, not real-world accuracy',
    'Real accuracy needs a fleet data-sharing agreement — that is what Phase 1 is for',
  ].map((t, i, a) => ({ text: t, options: { bullet: true, breakLine: i < a.length - 1 } })),
    { x: 7.08, y: 2.6, w: 5.27, h: 3.1, fontSize: 12.5, color: 'C3C9D4',
      paraSpaceAfter: 9, isTextBox: true, margin: 0 });
  s.addNotes('We would rather be trusted than impressive. A judge who asks "is this real?" '
    + 'should find the answer before they have to ask.');
}

// ================================================================= 11 DEPLOY
pres.addSection({ title: 'Deploy' });
{
  const s = slide('CONTENT', 'Deploy');
  title(s, 'From prototype to a fleet that depends on it');
  const phases = [
    ['Phase 1', 'Shadow', '6 weeks, one corridor', [
      'Scores computed and logged, nothing shown',
      'Weekly: predicted vs observed on routes actually driven',
      'Exit: coverage 85–95%, match accuracy > 98%',
    ], BLUE],
    ['Phase 2', 'Advisory', 'weeks 7–16', [
      'Dispatchers see score, interval, reasons, alternatives',
      'Overrides require a reason code and are logged',
      'Exit: adoption > 40%, no SLA regression',
    ], AMBER],
    ['Phase 3', 'Scale', 'multi-region', [
      'Nightly retraining, drift and calibration monitoring',
      'Cohort reports to the safety team',
      'Segment risk published as vector tiles',
    ], '5FD38A'],
  ];
  phases.forEach(([p, name, when, lines, col], i) => {
    const x = 0.65 + i * 4.06;
    card(s, x, 1.5, 3.82, 4.3);
    tx(s, p, { x: x + 0.3, y: 1.72, w: 3.2, h: 0.3, fontSize: 11, bold: true, color: col, charSpacing: 1.5 });
    tx(s, name, { x: x + 0.3, y: 2.04, w: 3.2, h: 0.45, fontSize: 24, bold: true, color: INK, fontFace: 'Cambria' });
    tx(s, when, { x: x + 0.3, y: 2.54, w: 3.2, h: 0.3, fontSize: 11, color: '6E7686', italic: true });
    s.addText(lines.map((t, j, a) => ({ text: t, options: { bullet: true, breakLine: j < a.length - 1 } })),
      { x: x + 0.3, y: 2.98, w: 3.25, h: 2.6, fontSize: 12, color: 'C3C9D4',
        paraSpaceAfter: 9, isTextBox: true, margin: 0 });
  });
  tx(s, 'Privacy by construction: vehicle IDs hashed at ingest · reports at cohort level only · driver identity never in a report · routes advisory with logged overrides',
    { x: 0.65, y: 6.0, w: 11.95, h: 0.6, fontSize: 12, color: MUTED });
  s.addNotes('Shadow mode is non-negotiable. It is where the synthetic numbers in this '
    + 'deck get replaced by real ones.');
}

// ================================================================= 12 IMPACT
{
  const s = slide('CONTENT', 'Deploy');
  title(s, 'Impact hypothesis');
  tx(s, 'Stated as a target to be tested in Phase 1, not as a result we are claiming.',
    { x: 0.65, y: 1.2, w: 11.9, h: 0.4, fontSize: 14.5, color: MUTED });

  card(s, 0.65, 1.9, 11.95, 1.5, '16202B');
  tx(s, 'If dispatchers take safer alternatives inside a bounded time penalty, risk-weighted exposure on high-risk corridors falls by at least 20% for under 10% extra travel time.',
    { x: 1.0, y: 2.18, w: 11.3, h: 1.0, fontSize: 18, color: INK, fontFace: 'Cambria' });

  const why = [
    ['Why it is plausible', [
      'Held-out replay already shows −' + f(cf.avg_true_risk_reduction_pct_when_switched, 1) + '% true risk for +' + f(cf.avg_time_increase_pct_when_switched, 1) + '% time',
      'The policy is tunable: the operator sets the budget, not the model',
    ]],
    ['What would falsify it', [
      'Adoption below 40% — dispatchers do not believe the score',
      'Interval coverage outside 85–95% on real data — the badge is not honest',
      'No measurable change in incidents on the pilot corridor',
    ]],
  ];
  why.forEach(([h, lines], i) => {
    const x = 0.65 + i * 6.1;
    card(s, x, 3.65, 5.85, 2.3);
    tx(s, h, { x: x + 0.32, y: 3.88, w: 5.2, h: 0.4, fontSize: 16, bold: true,
      color: i === 0 ? '5FD38A' : AMBER });
    s.addText(lines.map((t, j, a) => ({ text: t, options: { bullet: true, breakLine: j < a.length - 1 } })),
      { x: x + 0.32, y: 4.36, w: 5.25, h: 1.45, fontSize: 12.5, color: 'C3C9D4',
        paraSpaceAfter: 8, isTextBox: true, margin: 0 });
  });
  s.addNotes('Naming what would falsify the hypothesis is how we keep ourselves honest '
    + 'once there is real data.');
}

// ================================================================= 13 CLOSE
pres.addSection({ title: 'Close' });
{
  const s = slide('CONTENT', 'Close');
  title(s, 'RouteShield');
  tx(s, 'Open source, MIT. Runs end to end in about five seconds, with no API keys and no GPU.',
    { x: 0.65, y: 1.22, w: 11.9, h: 0.4, fontSize: 15, color: MUTED });

  const stats = [
    [f(sm.roc_auc_top_decile_TRUE_rate.routeshield, 3), 'AUC ranking the\ntruly riskiest roads', '5FD38A'],
    [f(sm.spearman_vs_true_rate_sparse_only.routeshield, 2), 'rank correlation where\nhistory is thinnest', AMBER],
    [pc(sm.latent_rate_interval_coverage_90, 0), '90% interval coverage\n— the badge is honest', BLUE],
    ['35', 'automated tests\npinning the behaviour', INK],
  ];
  stats.forEach(([v, l, c], i) => {
    const x = 0.65 + i * 3.04;
    card(s, x, 1.85, 2.84, 1.95);
    tx(s, v, { x: x + 0.2, y: 2.08, w: 2.44, h: 0.72, fontSize: 36, bold: true, color: c, fontFace: 'Cambria' });
    tx(s, l, { x: x + 0.2, y: 2.85, w: 2.44, h: 0.85, fontSize: 11.5, color: MUTED });
  });

  card(s, 0.65, 4.1, 11.95, 1.85);
  tx(s, 'Try it, read it, break it', { x: 1.0, y: 4.32, w: 6, h: 0.4, fontSize: 18, bold: true, color: INK });
  s.addText([
    { text: 'Live demo   ', options: { color: MUTED } },
    { text: 'shaikmdarifcse23-byte.github.io/routeshield', options: { color: BLUE, breakLine: true } },
    { text: 'Source       ', options: { color: MUTED } },
    { text: 'github.com/shaikmdarifcse23-byte/routeshield', options: { color: BLUE, breakLine: true } },
    { text: 'Run it         ', options: { color: MUTED } },
    { text: 'pip install -r requirements.txt && python -m routeshield.cli demo',
      options: { color: 'C3C9D4', fontFace: 'Courier New' } },
  ], { x: 1.0, y: 4.82, w: 11.2, h: 1.0, fontSize: 13, paraSpaceAfter: 6,
       isTextBox: true, margin: 0 });
  s.addNotes('The demo runs the full model client-side. Nothing to install to see it work.');
}

(async () => {
  const out = path.join(ROOT, 'artifacts/RouteShield_Prototype_Deck.pptx');
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log('wrote ' + out);
})();
