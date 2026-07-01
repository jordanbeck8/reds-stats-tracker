import * as Plot from "npm:@observablehq/plot";
import * as d3 from "npm:d3";

/**
 * Radar chart of players across normalized axes.
 * data: [{name, axis, value}] with value in [0, 1] (team-percentile normalized).
 */
export function radar(data, {width = 480} = {}) {
  const axes = [...new Set(data.map((d) => d.axis))];
  const names = [...new Set(data.map((d) => d.name))];
  const angle = d3.scalePoint(axes, [0, 2 * Math.PI - (2 * Math.PI) / axes.length]);
  const r = 0.42 * Math.min(width, 420);
  const cx = width / 2, cy = 210;

  const px = (axis, value) => cx + r * value * Math.cos(angle(axis) - Math.PI / 2);
  const py = (axis, value) => cy + r * value * Math.sin(angle(axis) - Math.PI / 2);

  const points = data.map((d) => ({...d, x: px(d.axis, d.value), y: py(d.axis, d.value)}));
  // Close each polygon by repeating its first point.
  const closed = names.flatMap((name) => {
    const own = points.filter((p) => p.name === name);
    return [...own, own[0]];
  });

  const rings = [0.25, 0.5, 0.75, 1].flatMap((v) =>
    [...axes, axes[0]].map((axis) => ({ring: v, axis, x: px(axis, v), y: py(axis, v)}))
  );

  return Plot.plot({
    width,
    height: 420,
    x: {axis: null},
    y: {axis: null},
    color: {legend: true, range: ["#c6011f", "#000000", "#6b7280", "#b91c1c"]},
    marks: [
      Plot.line(rings, {x: "x", y: "y", z: "ring", stroke: "#ddd", strokeWidth: 0.5}),
      ...axes.map((axis) =>
        Plot.link([axis], {x1: cx, y1: cy, x2: px(axis, 1), y2: py(axis, 1), stroke: "#ddd", strokeWidth: 0.5})
      ),
      Plot.text(axes.map((axis) => ({axis, x: px(axis, 1.14), y: py(axis, 1.14)})),
        {x: "x", y: "y", text: "axis", fontSize: 11, fontWeight: 600}),
      Plot.area(closed, {x1: "x", y1: "y", fill: "name", fillOpacity: 0.12, curve: "linear-closed"}),
      Plot.line(closed, {x: "x", y: "y", stroke: "name", strokeWidth: 2, curve: "linear-closed"}),
      Plot.dot(points, {x: "x", y: "y", fill: "name", r: 3, tip: true,
        title: (d) => `${d.name}\n${d.axis}: ${d.raw ?? d.value}`})
    ]
  });
}

/** Percentile of value within values (0..1), higher = better. */
export function percentile(value, values, invert = false) {
  const sorted = values.filter((v) => v != null).sort(d3.ascending);
  if (value == null || !sorted.length) return 0;
  const p = sorted.filter((v) => v <= value).length / sorted.length;
  return invert ? 1 - p + 1 / sorted.length : p;
}
