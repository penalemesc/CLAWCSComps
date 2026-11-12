#import "@preview/lilaq:0.6.0" as lq
#set page(paper: "a3")

#show: lq.show_(
  lq.tick-label.with(kind: "x"),
  it => box(
    width: 0pt,
    align(right, rotate(-45deg, reflow: true, it))
  ),
)

// Values reconstructed from accuracy.svg (rounded to three decimals).
// Model order: YOLO26m zero-shot, YOLO26m fine-tuned,
//              SAM 3 zero-shot, SAM 3 fine-tuned.
#let models = (0, 1, 2, 3)
#let box-ap = (0.463, 0.534, 0.532, 0.573)
#let ap50 = (0.662, 0.742, 0.767, 0.802)
#let small-ap = (0.222, 0.319, 0.319, 0.356)
#let f1 = (0.688, 0.755, 0.763, 0.787)

// Values from compare_miss (1).svg. Rename the right-axis label
// if the percentage has a more specific meaning in your experiment.
#let compare = (52, 67, 71, 81)

// Distinct colors for accuracy metrics.
#let c-box = rgb("#447D72")
#let c-ap50 = rgb("#79B8A3")
#let c-small = rgb("#DAA15F")
#let c-f1 = rgb("#8D7AB6")
#let c-compare = rgb("#292D35")

// Original green/pink model-family colors for the time chart.
#let c-yolo = rgb("#629E56")
#let c-sam = rgb("#BF6AA5")

  #lq.diagram(
    width: 17cm,
    height: 10cm,
    title: [Accuracy and comparison],
    ylabel: [Accuracy score],
    xlim: (-0.50, 3.50),
    ylim: (0, 0.90),
    xaxis: (
      ticks: (
        (0, [YOLO26m ZS]),
        (1, [YOLO26m FT]),
        (2, [SAM 3 ZS]),
        (3, [SAM 3 FT]),
      ),
      subticks: none,
    ),
    yaxis: (
      mirror: false,
      tick-distance: 0.1,
      subticks: none,
    ),
    grid: (stroke: 0.4pt + rgb("#E2E5EA")),
    legend: (position: (0%, 0%)),

    // Four metrics per model, each offset within its group.
    lq.bar(models, box-ap,
      offset: -0.27, width: 0.17,
      fill: c-box, label: [Box AP]),
    lq.bar(models, ap50,
      offset: -0.09, width: 0.17,
      fill: c-ap50, label: [AP50]),
    lq.bar(models, small-ap,
      offset: 0.09, width: 0.17,
      fill: c-small, label: [AP small cars]),
    lq.bar(models, f1,
      offset: 0.27, width: 0.17,
      fill: c-f1, label: [F1]),

    // Approximate uncertainty whiskers digitized from accuracy.svg.
    // These are only graphical reconstructions, not raw test statistics.
    lq.plot(models.map(x => x - 0.27), box-ap,
      yerr: (0.012, 0.013, 0.010, 0.010),
      color: c-compare, stroke: none, mark: none, z-index: 4),
    lq.plot(models.map(x => x - 0.09), ap50,
      yerr: (0.012, 0.012, 0.012, 0.009),
      color: c-compare, stroke: none, mark: none, z-index: 4),
    lq.plot(models.map(x => x + 0.09), small-ap,
      yerr: (0.012, 0.012, 0.012, 0.010),
      color: c-compare, stroke: none, mark: none, z-index: 4),
    lq.plot(models.map(x => x + 0.27), f1,
      yerr: (0.011, 0.011, 0.012, 0.009),
      color: c-compare, stroke: none, mark: none, z-index: 4),

    // Independent right-hand y-axis, not a rescaling of accuracy.
    lq.yaxis(
      position: right,
      lim: (0, 100),
      label: [Compare (%)],
      ticks: (0, 20, 40, 60, 80, 100),
      subticks: none,
      lq.plot(models, compare,
        color: c-compare,
        stroke: 1.7pt,
        mark: "o",
        mark-size: 5pt,
        z-index: 5,
        label: [$"Output Boxes" / "Ground Truth"$]),
    ),
  )
#v(3pt)
#text(size: 8pt, fill: rgb("#535966"))[ZS = zero-shot; FT = fine-tuned.]
