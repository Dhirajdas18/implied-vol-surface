// Surface maths in the browser. Mirrors src/iv/surface.py; tests compare the two through Node.
// An SVI slice is { a, b, rho, m, sigma } in Gatheral's raw parameterisation of total implied
// variance w = sigma_imp^2 T against log-moneyness k = ln(K / F).

// Total variance: w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)).
function sviW(p, k) {
  const x = k - p.m;
  return p.a + p.b * (p.rho * x + Math.sqrt(x * x + p.sigma * p.sigma));
}

// Implied volatility from total variance and time to expiry in years (0 below zero variance).
function ivFromW(w, T) {
  return Math.sqrt(Math.max(w, 0) / T);
}

function sviIv(p, k, T) {
  return ivFromW(sviW(p, k), T);
}

// Gatheral's butterfly function g(k); g < 0 means a negative implied density (arbitrage).
// g = (1 - k w' / (2w))^2 - (w'^2 / 4)(1/w + 1/4) + w'' / 2.
function sviG(p, k) {
  const x = k - p.m;
  const root = Math.sqrt(x * x + p.sigma * p.sigma);
  const w = p.a + p.b * (p.rho * x + root);
  const w1 = p.b * (p.rho + x / root);
  const w2 = p.b * p.sigma * p.sigma / (root * root * root);
  return (1 - k * w1 / (2 * w)) ** 2 - (w1 * w1 / 4) * (1 / w + 0.25) + w2 / 2;
}

if (typeof module !== "undefined") {
  module.exports = { sviW, ivFromW, sviIv, sviG };
}
