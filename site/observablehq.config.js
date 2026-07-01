export default {
  title: "Reds Dashboard",
  pages: [
    {name: "Overview", path: "/"},
    {name: "Hitting", path: "/hitting"},
    {name: "Pitching", path: "/pitching"},
    {name: "Fielding", path: "/fielding"},
    {name: "Compare Players", path: "/compare"},
    {name: "Trends", path: "/trends"}
  ],
  theme: "light",
  style: "style.css",
  head: `
    <link rel="preload" as="font" type="font/woff2" href="/fonts/montserrat-v31-latin-regular.woff2" crossorigin>
    <link rel="preload" as="font" type="font/woff2" href="/fonts/montserrat-v31-latin-600.woff2" crossorigin>
    <link rel="preload" as="font" type="font/woff2" href="/fonts/montserrat-v31-latin-800.woff2" crossorigin>
    <style>
      @font-face { font-family: "Montserrat"; font-style: normal; font-weight: 400; font-display: swap; src: url("/fonts/montserrat-v31-latin-regular.woff2") format("woff2"); }
      @font-face { font-family: "Montserrat"; font-style: normal; font-weight: 600; font-display: swap; src: url("/fonts/montserrat-v31-latin-600.woff2") format("woff2"); }
      @font-face { font-family: "Montserrat"; font-style: normal; font-weight: 800; font-display: swap; src: url("/fonts/montserrat-v31-latin-800.woff2") format("woff2"); }
    </style>
  `,
  root: "src",
  cleanUrls: false, // plain .html links — works under tailscale serve / any static server
  toc: false,
  pager: false,
  footer: `Data: MLB Stats API · Baseball-Reference (bWAR) · FanGraphs (fWAR) — not affiliated with MLB or the Cincinnati Reds.`
};
