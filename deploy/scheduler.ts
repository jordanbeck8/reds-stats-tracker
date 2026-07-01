#!/usr/bin/env bun
/**
 * Post-game scheduler — creates a one-shot systemd timer per Reds game today,
 * firing reds-update.service at game start + 3h.
 *
 * Runs twice daily (09:00 & 16:00 ET) via reds-scheduler.timer to catch
 * schedule changes, postponements, and doubleheaders. Transient units
 * self-clean after firing; the nightly midnight run self-heals any miss.
 *
 * Usage: bun deploy/scheduler.ts [--dry-run]
 */

const TEAM_ID = 113;
const DELAY_MS = 3 * 60 * 60 * 1000; // 3 hours after first pitch
const DRY_RUN = process.argv.includes("--dry-run");

const todayET = new Date().toLocaleDateString("en-CA", {timeZone: "America/New_York"});

interface Game {
  gamePk: number;
  gameDate: string;
  status: {abstractGameState: string};
}

async function todaysGames(): Promise<Game[]> {
  const url =
    `https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId=${TEAM_ID}` +
    `&startDate=${todayET}&endDate=${todayET}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`schedule fetch failed: ${resp.status}`);
  const data = await resp.json();
  return (data.dates ?? []).flatMap((d: {games: Game[]}) => d.games);
}

async function existingTimers(): Promise<Set<string>> {
  const proc = Bun.spawn(["systemctl", "--user", "list-timers", "--all", "--output=json"]);
  const out = await new Response(proc.stdout).text();
  try {
    return new Set(JSON.parse(out).map((t: {unit: string}) => t.unit));
  } catch {
    return new Set();
  }
}

function fmtLocal(d: Date): string {
  // systemd-run --on-calendar wants a local "YYYY-MM-DD HH:MM:SS".
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
         `${p(d.getHours())}:${p(d.getMinutes())}:00`;
}

const games = await todaysGames();
if (!games.length) {
  console.log(`No Reds game today (${todayET}).`);
  process.exit(0);
}

const timers = DRY_RUN ? new Set<string>() : await existingTimers();

for (const g of games) {
  const unit = `reds-postgame-${g.gamePk}`;
  const final = g.status.abstractGameState === "Final";
  // Final already? Update in 2 minutes. Otherwise first pitch + 3h.
  const target = final
    ? new Date(Date.now() + 2 * 60 * 1000)
    : new Date(new Date(g.gameDate).getTime() + DELAY_MS);

  if (target.getTime() < Date.now() && !final) {
    console.log(`skip ${unit}: target ${fmtLocal(target)} already past (midnight run covers it)`);
    continue;
  }
  if (timers.has(`${unit}.timer`)) {
    console.log(`skip ${unit}: timer already armed`);
    continue;
  }

  const cmd = [
    "systemd-run", "--user", `--unit=${unit}`,
    `--on-calendar=${fmtLocal(target)}`,
    "--timer-property=AccuracySec=1min",
    "systemctl", "--user", "start", "reds-update.service",
  ];

  if (DRY_RUN) {
    console.log(`[dry-run] ${cmd.join(" ")}`);
  } else {
    const proc = Bun.spawn(cmd, {stderr: "pipe"});
    if ((await proc.exited) !== 0) {
      console.error(`FAILED ${unit}: ${await new Response(proc.stderr).text()}`);
    } else {
      console.log(`armed ${unit} @ ${fmtLocal(target)}`);
    }
  }
}
