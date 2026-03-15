#!/usr/bin/env node
/**
 * Exotic Options Benchmark
 *
 * Benchmarks all three exotic options analysis pipelines across multiple data sizes:
 *   1. Hyperbolic Options Discovery (encoding, GNN contagion, HNSW regime matching)
 *   2. Topological Vol Analysis (persistent homology, Ricci flow)
 *   3. Full end-to-end pipelines
 *
 * Run: node examples/neural-trader/exotic/benchmark-options.js
 */

import { performance } from 'perf_hooks';

import {
  HyperbolicOptionsDiscovery,
  OptionsChainGenerator,
  OptionsHyperbolicEncoder,
  VolSurfaceGNN,
  VolRegimeHNSW,
  discoveryConfig
} from './hyperbolic-options-discovery.js';

import {
  TopologicalVolAnalysis,
  VolSurfacePersistence,
  VolSurfaceRicciFlow,
  tdaConfig
} from './topological-vol-analysis.js';

// =============================================================================
// CONFIGURATION
// =============================================================================

const ITERATIONS = 5;

const DATA_SIZES = [
  { label: 'small',  numStrikes: 4,  numExpiries: 3 },
  { label: 'medium', numStrikes: 8,  numExpiries: 4 },
  { label: 'large',  numStrikes: 12, numExpiries: 6 },
];

// =============================================================================
// HELPERS
// =============================================================================

function formatMs(ms) {
  return ms < 1 ? `${(ms * 1000).toFixed(0)}us` : `${ms.toFixed(2)}ms`;
}

function formatOps(ms) {
  if (ms <= 0) return 'Inf';
  return (1000 / ms).toFixed(1);
}

function memDiffMB(before, after) {
  return ((after.heapUsed - before.heapUsed) / 1024 / 1024).toFixed(2);
}

function printTable(headers, rows) {
  const colWidths = headers.map((h, i) => {
    const maxData = rows.reduce((max, r) => Math.max(max, String(r[i]).length), 0);
    return Math.max(h.length, maxData) + 2;
  });

  const sep = colWidths.map(w => '-'.repeat(w)).join('+');
  const line = (cols) => cols.map((c, i) => String(c).padEnd(colWidths[i])).join('|');

  console.log(sep);
  console.log(line(headers));
  console.log(sep);
  for (const row of rows) {
    console.log(line(row));
  }
  console.log(sep);
}

function generateChain(size) {
  return OptionsChainGenerator.generate('BENCH', 100, {
    numStrikes: size.numStrikes,
    numExpiries: size.numExpiries,
    baseIV: 25,
    anomalyCount: 2,
    skew: -0.15
  });
}

// =============================================================================
// INDIVIDUAL COMPONENT BENCHMARKS
// =============================================================================

async function benchmarkComponent(name, setupFn, runFn, chain, iterations) {
  const times = [];
  let ctx;
  for (let i = 0; i < iterations; i++) {
    ctx = setupFn();
    const t0 = performance.now();
    runFn(ctx, chain);
    const t1 = performance.now();
    times.push(t1 - t0);
  }
  const min = Math.min(...times);
  const max = Math.max(...times);
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  return { name, min, max, avg, opsPerSec: 1000 / avg };
}

async function benchmarkHyperbolicEncoding(chain, iterations) {
  return benchmarkComponent(
    'Hyperbolic Encoding',
    () => new OptionsHyperbolicEncoder(discoveryConfig),
    (encoder, ch) => encoder.encode(ch),
    chain,
    iterations
  );
}

async function benchmarkGNNContagion(chain, iterations) {
  // Need an encoding first for GNN
  const encoder = new OptionsHyperbolicEncoder(discoveryConfig);
  const encoding = encoder.encode(chain);

  return benchmarkComponent(
    'GNN Contagion',
    () => new VolSurfaceGNN(discoveryConfig.gnn),
    (gnn, ch) => {
      gnn.buildGraph(ch, encoding.embeddings);
      // Find seed nodes (high volume options)
      const avgVol = ch.options.reduce((s, o) => s + (o.volume || 0), 0) / ch.options.length;
      const seeds = ch.options
        .filter(o => (o.volume || 0) > avgVol * 2)
        .slice(0, 3)
        .map(o => `${o.type}_${o.strike}_${o.expiry}`);
      if (seeds.length > 0) {
        gnn.detectContagion(seeds);
      }
    },
    chain,
    iterations
  );
}

async function benchmarkHNSWRegime(chain, iterations) {
  const matcher = new VolRegimeHNSW(discoveryConfig.hnsw);
  await matcher.init(null); // JS fallback

  return benchmarkComponent(
    'HNSW Regime Match',
    () => matcher,
    (m, ch) => m.matchRegime(ch),
    chain,
    iterations
  );
}

async function benchmarkPersistentHomology(chain, iterations) {
  return benchmarkComponent(
    'Persistent Homology',
    () => new VolSurfacePersistence(tdaConfig.homology),
    (persistence, ch) => {
      persistence.buildComplex(ch);
      persistence.computePersistence();
      persistence.extractSignals();
    },
    chain,
    iterations
  );
}

async function benchmarkRicciFlow(chain, iterations) {
  return benchmarkComponent(
    'Ricci Flow',
    () => new VolSurfaceRicciFlow(tdaConfig.ricciFlow),
    (ricci, ch) => {
      ricci.buildGraph(ch);
      ricci.flow();
      ricci.extractSignals();
    },
    chain,
    iterations
  );
}

async function benchmarkFullHyperbolicPipeline(chain, iterations) {
  const discovery = new HyperbolicOptionsDiscovery(discoveryConfig);
  await discovery.init(null);

  return benchmarkComponent(
    'Full Hyperbolic Pipeline',
    () => discovery,
    (d, ch) => {
      // Reset internal state for fresh run
      d.encoder = new OptionsHyperbolicEncoder(discoveryConfig);
      d.gnn = new VolSurfaceGNN(discoveryConfig.gnn);
      d.analyze(ch, 100000);
    },
    chain,
    iterations
  );
}

async function benchmarkFullTopologicalPipeline(chain, iterations) {
  return benchmarkComponent(
    'Full Topological Pipeline',
    () => new TopologicalVolAnalysis(tdaConfig),
    (topo, ch) => topo.analyze(ch),
    chain,
    iterations
  );
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  console.log('='.repeat(80));
  console.log('  EXOTIC OPTIONS BENCHMARK');
  console.log('  Iterations per benchmark:', ITERATIONS);
  console.log('='.repeat(80));
  console.log();

  const allResults = [];

  for (const size of DATA_SIZES) {
    console.log(`\n${'─'.repeat(80)}`);
    console.log(`  DATA SIZE: ${size.label.toUpperCase()} (${size.numStrikes} strikes x ${size.numExpiries} expiries)`);
    console.log(`${'─'.repeat(80)}`);

    const chain = generateChain(size);
    console.log(`  Generated ${chain.options.length} options\n`);

    // Memory snapshot before
    global.gc?.();
    const memBefore = process.memoryUsage();

    // Run all component benchmarks
    const results = [];
    results.push(await benchmarkHyperbolicEncoding(chain, ITERATIONS));
    results.push(await benchmarkGNNContagion(chain, ITERATIONS));
    results.push(await benchmarkHNSWRegime(chain, ITERATIONS));
    results.push(await benchmarkPersistentHomology(chain, ITERATIONS));
    results.push(await benchmarkRicciFlow(chain, ITERATIONS));
    results.push(await benchmarkFullHyperbolicPipeline(chain, ITERATIONS));
    results.push(await benchmarkFullTopologicalPipeline(chain, ITERATIONS));

    // Memory snapshot after
    const memAfter = process.memoryUsage();

    // Print component results table
    const headers = ['Component', 'Min', 'Avg', 'Max', 'Ops/sec'];
    const rows = results.map(r => [
      r.name,
      formatMs(r.min),
      formatMs(r.avg),
      formatMs(r.max),
      formatOps(r.avg)
    ]);

    printTable(headers, rows);

    // Memory usage
    console.log(`\n  Memory Usage:`);
    console.log(`    Heap delta: ${memDiffMB(memBefore, memAfter)} MB`);
    console.log(`    RSS:        ${(memAfter.rss / 1024 / 1024).toFixed(2)} MB`);
    console.log(`    Heap used:  ${(memAfter.heapUsed / 1024 / 1024).toFixed(2)} MB`);
    console.log(`    Heap total: ${(memAfter.heapTotal / 1024 / 1024).toFixed(2)} MB`);

    // Bottleneck analysis (component-level only, exclude full pipelines)
    const componentResults = results.filter(r => !r.name.startsWith('Full'));
    const totalComponentTime = componentResults.reduce((s, r) => s + r.avg, 0);
    const sorted = [...componentResults].sort((a, b) => b.avg - a.avg);

    console.log(`\n  Bottleneck Analysis (component avg times):`);
    for (const r of sorted) {
      const pct = ((r.avg / totalComponentTime) * 100).toFixed(1);
      const bar = '#'.repeat(Math.round(Number(pct) / 2));
      console.log(`    ${r.name.padEnd(24)} ${formatMs(r.avg).padEnd(12)} ${pct.padStart(5)}%  ${bar}`);
    }
    console.log(`    ${'TOTAL'.padEnd(24)} ${formatMs(totalComponentTime)}`);

    const slowest = sorted[0];
    console.log(`\n  >> Slowest component: ${slowest.name} (${formatMs(slowest.avg)}, ${((slowest.avg / totalComponentTime) * 100).toFixed(1)}% of total)`);

    allResults.push({ size: size.label, chain, results, memBefore, memAfter });
  }

  // =============================================================================
  // CROSS-SIZE COMPARISON
  // =============================================================================

  console.log(`\n\n${'='.repeat(80)}`);
  console.log('  SCALING ANALYSIS (avg ms across data sizes)');
  console.log(`${'='.repeat(80)}\n`);

  const componentNames = allResults[0].results.map(r => r.name);
  const scaleHeaders = ['Component', ...DATA_SIZES.map(s => `${s.label} (${s.numStrikes}x${s.numExpiries})`)];
  const scaleRows = componentNames.map(name => {
    const row = [name];
    for (const sizeResult of allResults) {
      const r = sizeResult.results.find(r => r.name === name);
      row.push(formatMs(r.avg));
    }
    return row;
  });

  printTable(scaleHeaders, scaleRows);

  console.log('\nBenchmark complete.\n');
}

main().catch(err => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
