import {html} from "npm:htl";

const fmtDate = (iso) =>
  new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", {
    weekday: "short", month: "short", day: "numeric", timeZone: "UTC"
  });

const fmtTime = (utc) =>
  new Date(utc).toLocaleTimeString("en-US", {
    hour: "numeric", minute: "2-digit", timeZone: "America/New_York"
  });

function gameCard(g, todayISO) {
  const redsScore = g.isRedsHome ? g.homeScore : g.awayScore;
  const oppScore = g.isRedsHome ? g.awayScore : g.homeScore;
  const isFinal = g.state.startsWith("Final") || g.state === "Completed Early" || g.state === "Game Over";

  let result;
  if (isFinal && redsScore != null) {
    const won = redsScore > oppScore;
    result = html`<div class="cal-result ${won ? "win" : "loss"}">${won ? "W" : "L"} ${redsScore}-${oppScore}</div>`;
  } else if (g.state === "Postponed") {
    result = html`<div class="cal-result upcoming">PPD</div>`;
  } else {
    result = html`<div class="cal-result upcoming">${fmtTime(g.gameDateUTC)}</div>`;
  }

  const logo = html`<img src="https://www.mlbstatic.com/team-logos/${g.opponentId}.svg"
    alt="${g.opponentAbbr}" loading="lazy" onerror=${(e) => e.currentTarget.remove()}>`;

  return html`<div class="cal-game ${g.date === todayISO ? "cal-today" : ""}" data-date=${g.date}>
    <div class="cal-date">${fmtDate(g.date)}</div>
    <div class="cal-opp">${g.isRedsHome ? "vs" : "@"} ${logo} ${g.opponentAbbr}</div>
    ${result}
  </div>`;
}

export function CalendarBar(schedule) {
  const todayISO = new Date().toLocaleDateString("en-CA", {timeZone: "America/New_York"});
  const bar = html`<div class="cal-bar">${schedule.map((g) => gameCard(g, todayISO))}</div>`;

  // Center today's (or the next upcoming) game once the bar is in the DOM.
  requestAnimationFrame(() => {
    const target =
      bar.querySelector(".cal-today") ??
      bar.querySelector(`[data-date="${schedule.find((g) => g.date >= todayISO)?.date}"]`);
    if (target) {
      bar.scrollLeft = target.offsetLeft - bar.clientWidth / 2 + target.clientWidth / 2;
    }
  });

  return bar;
}

export function teamRecord(schedule) {
  let w = 0, l = 0;
  for (const g of schedule) {
    if (!g.state.startsWith("Final") || g.homeScore == null) continue;
    const redsScore = g.isRedsHome ? g.homeScore : g.awayScore;
    const oppScore = g.isRedsHome ? g.awayScore : g.homeScore;
    redsScore > oppScore ? w++ : l++;
  }
  return {w, l};
}
