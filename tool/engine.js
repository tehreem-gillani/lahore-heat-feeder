// Browser version of the study (same rules as src/scooter.py and src/transformer.py).
// Differences from the Python run: the feeder load is stored as 10-minute means
// (held flat inside each 10 minutes) and the random plug-in times come from a
// different generator, so results differ slightly from the report.
(function (root) {
  const PF = 0.95;
  const IEC = { x: 0.8, y: 1.6, k11: 1, k21: 1, k22: 2, tauO: 180, tauW: 4, R: 5, dOr: 55, dHr: 23 };
  const WH_PER_KM = 21.3, ETA = 0.85;
  const PROFILES = {
    commuter: { km: 24, days: [0, 1, 2, 3, 4, 5], plug: [18, 19.5], leave: 7.75 },
    delivery: { km: 96, days: [0, 1, 2, 3, 4, 5, 6], plug: [22.5, 23.5], leave: 10.75 },
  };
  const PEAK_END = { 12: 21, 1: 21, 2: 21, 3: 22, 4: 22, 5: 22, 6: 23, 7: 23, 8: 23, 9: 22, 10: 22, 11: 22 };
  const DAY = 1440;

  function rng(seed) {             // mulberry32
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gridKwh(kind, factor) { return PROFILES[kind].km * WH_PER_KM * factor / 1000 / ETA; }

  function charging(nMin, startDate, p) {
    const n = Math.round(p.adoption * p.houses);
    const nDel = Math.round(n * p.shareDelivery);
    const r = rng(p.seed || 1);
    const fleet = [];
    for (let i = 0; i < n; i++) {
      const kind = i < nDel ? "delivery" : "commuter";
      const w = PROFILES[kind].plug;
      fleet.push({ kind, plug: w[0] + r() * (w[1] - w[0]) });
    }
    const load = new Float64Array(nMin);
    const nDays = Math.floor(nMin / DAY);
    for (let d = 0; d < nDays; d++) {
      const date = new Date(startDate.getTime() + d * 86400000);
      const wd = (date.getUTCDay() + 6) % 7;          // Monday = 0
      const month = date.getUTCMonth() + 1;
      for (const s of fleet) {
        const pr = PROFILES[s.kind];
        if (!pr.days.includes(wd)) continue;
        const need = Math.ceil(gridKwh(s.kind, p.energyFactor) / p.chargerKw * 60);
        const iPlug = d * DAY + Math.round(s.plug * 60);
        const iLeave = (d + 1) * DAY + Math.round(pr.leave * 60);
        if (iLeave > nMin) continue;
        let start;
        if (p.rule === "plug-in") start = iPlug;
        else if (p.rule === "after-peak") start = Math.max(iPlug, d * DAY + PEAK_END[month] * 60);
        else if (p.rule === "latest") start = Math.max(iPlug, iLeave - need);
        else throw new Error("unknown rule " + p.rule);
        const stop = Math.min(start + need, nMin);
        for (let i = start; i < stop; i++) load[i] += p.chargerKw;
      }
    }
    return load;
  }

  // IEC 60076-7 Annex C difference equations, 1-minute step, two-day warm-up.
  function transformer(pNet, tAir, sRated) {
    const n = pNet.length, w = 2 * DAY;
    const K = (i) => Math.abs(pNet[i]) / PF / sRated;
    const k0 = K(0);
    let to = tAir[0] + IEC.dOr * Math.pow((1 + k0 * k0 * IEC.R) / (1 + IEC.R), IEC.x);
    let h1 = IEC.k21 * IEC.dHr * Math.pow(k0, IEC.y);
    let h2 = (IEC.k21 - 1) * IEC.dHr * Math.pow(k0, IEC.y);
    const aO = 1 / (IEC.k11 * IEC.tauO), a1 = 1 / (IEC.k22 * IEC.tauW), a2 = 1 / (IEC.tauO / IEC.k22);
    const hs = new Float32Array(n);
    let lol = 0, hsMax = -1e9, kMax = 0, overMin = 0, over120 = 0;
    for (let j = 0; j < w + n; j++) {
      const i = j < w ? j : j - w;
      const k = K(i);
      to += aO * (IEC.dOr * Math.pow((1 + k * k * IEC.R) / (1 + IEC.R), IEC.x) - (to - tAir[i]));
      const ky = Math.pow(k, IEC.y);
      h1 += a1 * (IEC.k21 * IEC.dHr * ky - h1);
      h2 += a2 * ((IEC.k21 - 1) * IEC.dHr * ky - h2);
      if (j < w) continue;
      const th = to + h1 - h2;
      hs[i] = th;
      lol += Math.pow(2, (th - 98) / 6) / 60;
      if (th > hsMax) hsMax = th;
      if (k > kMax) kMax = k;
      if (k > 1) overMin++;
      if (th > 120) over120++;
    }
    return { hs, lol, hsMax, kMax, hoursOver: overMin / 60, hours120: over120 / 60 };
  }

  function expand(data) {
    const nMin = data.base10.length * 10;
    const base = new Float64Array(nMin);
    for (let i = 0; i < nMin; i++) base[i] = data.base10[Math.floor(i / 10)] / 10;
    const tAir = new Float64Array(nMin);
    const T = data.temp, nh = T.length;
    // each hourly value is the mean of the hour after its time stamp (POWER FAQ),
    // so it sits at mid-hour; the first and last half hours take the nearest value
    for (let i = 0; i < nMin; i++) {
      const h = Math.max(0, i / 60 - 0.5), h0 = Math.floor(h), f = h - h0;
      const a = T[Math.min(h0, nh - 1)] / 10, b = T[Math.min(h0 + 1, nh - 1)] / 10;
      tAir[i] = a + (b - a) * f;
    }
    return { base, tAir, start: new Date(data.start + "T00:00:00Z") };
  }

  function run(ex, p) {
    const ch = charging(ex.base.length, ex.start, p);
    const net = new Float64Array(ex.base.length);
    for (let i = 0; i < net.length; i++) net[i] = ex.base[i] + ch[i];
    const withS = transformer(net, ex.tAir, p.sRated);
    const without = transformer(ex.base, ex.tAir, p.sRated);
    let e = 0;
    for (let i = 0; i < ch.length; i++) e += ch[i] / 60;
    return { ch, withS, without, chargingMwh: e / 1000, ratio: withS.lol / without.lol };
  }

  root.FeederEngine = { expand, run, charging, transformer, PROFILES, PEAK_END };
})(typeof window !== "undefined" ? window : globalThis);
