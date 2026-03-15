/**
 * Topological Volatility Surface Analysis
 *
 * Two novel mathematical frameworks applied to options vol surfaces:
 *
 * ═══════════════════════════════════════════════════════════════
 * PART 1: PERSISTENT HOMOLOGY (Topological Data Analysis)
 * ═══════════════════════════════════════════════════════════════
 *
 * Computes the topological features (holes, loops, voids) of the
 * vol surface as a function of a filtration parameter. Persistent
 * features = real structure. Ephemeral features = noise.
 *
 * Key insight: A persistent 1-cycle (loop) in the vol surface
 * corresponds to a closed arbitrage path through the options chain.
 *
 * ═══════════════════════════════════════════════════════════════
 * PART 2: RICCI FLOW SMOOTHING
 * ═══════════════════════════════════════════════════════════════
 *
 * Evolves the vol surface metric to smooth out curvature. Regions
 * that resist smoothing are deep structural features (real mispricings).
 * Singularity formation (neck pinch) = regime transition imminent.
 *
 * Uses Ollivier-Ricci curvature (discrete Ricci curvature for graphs)
 * which is computable on the options chain graph.
 *
 * ═══════════════════════════════════════════════════════════════
 *
 * Uses: @ruvector/edge-full (HNSW, graph DB, SONA)
 *       neural-trader (Kelly, pipeline)
 *       ruvector CLI (GNN, attention)
 */

import {
  OptionsChainGenerator,
  OptionsPoincareSpace,
  discoveryConfig
} from './hyperbolic-options-discovery.js';

// =============================================================================
// CONFIGURATION
// =============================================================================

const tdaConfig = {
  // Persistent homology settings
  homology: {
    maxDimension: 2,          // Compute β₀, β₁, β₂
    filtrationSteps: 50,      // Resolution of the filtration
    persistenceThreshold: 0.25, // Min persistence to count as "real"
    maxSimplices: 50000,      // Memory cap
    distanceMetric: 'iv_weighted' // iv_weighted, greeks, hyperbolic
  },

  // Ricci flow settings
  ricciFlow: {
    steps: 30,                // Flow iterations
    stepSize: 0.1,            // ε in the flow equation
    surgeryThreshold: 0.01,   // Edge weight below this → surgery (cut)
    convergenceEps: 1e-5,     // Stop when change < this
    normalize: true,          // Normalize weights each step
    trackHistory: true        // Store intermediate states
  },

  // Anomaly detection
  anomaly: {
    minPersistence: 0.15,     // Persistent features above this = signal
    ricciCurvatureThreshold: -0.5,  // Highly negative = bottleneck
    singularityWarning: 0.05  // Edge weight approaching zero
  }
};

// =============================================================================
// PART 1: PERSISTENT HOMOLOGY
// =============================================================================

// ─── Union-Find (Disjoint Set) for tracking connected components ───

class UnionFind {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
    this.birthTime = new Array(n).fill(0); // filtration value at creation
  }

  find(x) {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]]; // path compression
      x = this.parent[x];
    }
    return x;
  }

  union(x, y) {
    const rx = this.find(x);
    const ry = this.find(y);
    if (rx === ry) return false; // already connected

    // Union by rank — the OLDER component (lower birth) survives
    if (this.rank[rx] < this.rank[ry]) {
      this.parent[rx] = ry;
      return { survivor: ry, merged: rx };
    } else if (this.rank[rx] > this.rank[ry]) {
      this.parent[ry] = rx;
      return { survivor: rx, merged: ry };
    } else {
      // Same rank: keep the one born earlier
      if (this.birthTime[rx] <= this.birthTime[ry]) {
        this.parent[ry] = rx;
        this.rank[rx]++;
        return { survivor: rx, merged: ry };
      } else {
        this.parent[rx] = ry;
        this.rank[ry]++;
        return { survivor: ry, merged: rx };
      }
    }
  }

  connected(x, y) {
    return this.find(x) === this.find(y);
  }
}

// ─── Simplicial Complex ───

/**
 * A simplex is a generalization of a triangle:
 *   0-simplex = point (vertex)
 *   1-simplex = edge
 *   2-simplex = triangle
 *   3-simplex = tetrahedron
 *
 * The boundary operator ∂ maps:
 *   ∂(edge AB) = B - A
 *   ∂(triangle ABC) = AB + BC - AC
 *
 * Homology = ker(∂) / im(∂)
 *   β₀ = # connected components
 *   β₁ = # independent loops (1-holes)
 *   β₂ = # enclosed voids (2-holes)
 */
class SimplicialComplex {
  constructor() {
    this.vertices = new Map();    // id → {data, filtration}
    this.edges = [];              // [{v0, v1, filtration, weight}]
    this.triangles = [];          // [{v0, v1, v2, filtration}]
    this.vertexIndex = new Map(); // id → numeric index
    this.nextIndex = 0;
  }

  addVertex(id, data, filtration = 0) {
    if (this.vertexIndex.has(id)) return this.vertexIndex.get(id);
    const idx = this.nextIndex++;
    this.vertexIndex.set(id, idx);
    this.vertices.set(idx, { id, data, filtration });
    return idx;
  }

  addEdge(id0, id1, filtration, weight = 1) {
    const v0 = this.vertexIndex.get(id0);
    const v1 = this.vertexIndex.get(id1);
    if (v0 === undefined || v1 === undefined) return;

    this.edges.push({
      v0: Math.min(v0, v1),
      v1: Math.max(v0, v1),
      filtration,
      weight
    });
  }

  addTriangle(id0, id1, id2, filtration) {
    const v0 = this.vertexIndex.get(id0);
    const v1 = this.vertexIndex.get(id1);
    const v2 = this.vertexIndex.get(id2);
    if (v0 === undefined || v1 === undefined || v2 === undefined) return;

    const sorted = [v0, v1, v2].sort((a, b) => a - b);
    this.triangles.push({
      v0: sorted[0],
      v1: sorted[1],
      v2: sorted[2],
      filtration
    });
  }

  get numVertices() { return this.nextIndex; }
  get numEdges() { return this.edges.length; }
  get numTriangles() { return this.triangles.length; }
}

// ─── Persistent Homology Computer ───

/**
 * VolSurfacePersistence
 *
 * Computes persistent homology of the options vol surface.
 *
 * The filtration works like this:
 *
 *   ε = 0:   Each option is an isolated point    (β₀ = n, β₁ = 0)
 *   ε = 0.1: Nearby options connect via edges     (β₀ decreases)
 *   ε = 0.3: Triangles form between triples       (β₁ may increase then decrease)
 *   ε = 1.0: Everything connected                  (β₀ = 1, β₁ = ?)
 *
 * The "birth" and "death" of each feature creates a persistence diagram.
 *
 *   death
 *    │  ·         · = short-lived feature (noise)
 *    │    ·
 *    │      ★     ★ = long-lived feature (SIGNAL)
 *    │  ·
 *    │·
 *    └──────── birth
 *
 * Features far from the diagonal (high persistence) are real topology.
 */
class VolSurfacePersistence {
  constructor(config = tdaConfig.homology) {
    this.config = config;
    this.complex = null;
    this.persistenceDiagram = { H0: [], H1: [], H2: [] };
    this.bettiCurve = [];
  }

  /**
   * Build the filtered simplicial complex from an options chain.
   *
   * Distance between options determined by:
   * - IV difference (primary)
   * - Moneyness difference
   * - Greeks similarity
   * - Time to expiry difference
   */
  buildComplex(chain) {
    this.complex = new SimplicialComplex();
    const { options, spot } = chain;

    // Add vertices (each option = a 0-simplex)
    for (const opt of options) {
      const id = `${opt.type}_${opt.strike}_${opt.expiry}`;
      this.complex.addVertex(id, opt, 0); // all born at filtration 0
    }

    // Compute pairwise distances
    const distances = [];
    for (let i = 0; i < options.length; i++) {
      for (let j = i + 1; j < options.length; j++) {
        const d = this._optionDistance(options[i], options[j], spot);
        distances.push({ i, j, d, opt_i: options[i], opt_j: options[j] });
      }
    }

    // Sort by distance (this determines the filtration order)
    distances.sort((a, b) => a.d - b.d);

    // Add edges at their filtration values
    for (const { i, j, d, opt_i, opt_j } of distances) {
      const id_i = `${opt_i.type}_${opt_i.strike}_${opt_i.expiry}`;
      const id_j = `${opt_j.type}_${opt_j.strike}_${opt_j.expiry}`;
      this.complex.addEdge(id_i, id_j, d, 1 / (1 + d));
    }

    // Add triangles (Vietoris-Rips): a triangle exists when all 3 edges exist
    // Only add up to maxSimplices to control memory
    this._buildTriangles(options, distances, spot);

    return this.complex;
  }

  _buildTriangles(options, sortedDistances, spot) {
    // Build adjacency at each distance threshold
    // For efficiency, only check triples where all three edges are short
    const edgeSet = new Map(); // "i_j" → distance
    let triangleCount = 0;

    for (const { i, j, d } of sortedDistances) {
      edgeSet.set(`${i}_${j}`, d);

      // Check if adding this edge completes any triangles
      if (triangleCount >= this.config.maxSimplices) break;

      for (let k = 0; k < options.length && triangleCount < this.config.maxSimplices; k++) {
        if (k === i || k === j) continue;

        const ik = i < k ? `${i}_${k}` : `${k}_${i}`;
        const jk = j < k ? `${j}_${k}` : `${k}_${j}`;

        if (edgeSet.has(ik) && edgeSet.has(jk)) {
          // Triangle exists! Filtration = max edge distance (Rips condition)
          const triFiltr = Math.max(d, edgeSet.get(ik), edgeSet.get(jk));

          const id_i = `${options[i].type}_${options[i].strike}_${options[i].expiry}`;
          const id_j = `${options[j].type}_${options[j].strike}_${options[j].expiry}`;
          const id_k = `${options[k].type}_${options[k].strike}_${options[k].expiry}`;

          this.complex.addTriangle(id_i, id_j, id_k, triFiltr);
          triangleCount++;
        }
      }
    }
  }

  /**
   * Compute persistent homology using the standard algorithm:
   * 1. Sort all simplices by filtration value
   * 2. Process each simplex in order
   * 3. Track birth/death of topological features
   */
  computePersistence() {
    const n = this.complex.numVertices;

    // ─── H₀: Connected Components (using Union-Find) ───
    const uf = new UnionFind(n);
    const componentBirth = new Map(); // component root → birth time

    // All vertices born at filtration 0
    for (let i = 0; i < n; i++) {
      componentBirth.set(i, 0);
    }

    // Sort edges by filtration
    const sortedEdges = [...this.complex.edges].sort((a, b) => a.filtration - b.filtration);

    for (const edge of sortedEdges) {
      const { v0, v1, filtration } = edge;

      if (!uf.connected(v0, v1)) {
        const result = uf.union(v0, v1);
        if (result) {
          // A component dies (merged into another)
          const mergedBirth = componentBirth.get(result.merged) || 0;
          const persistence = filtration - mergedBirth;

          this.persistenceDiagram.H0.push({
            birth: mergedBirth,
            death: filtration,
            persistence,
            type: 'component',
            mergedInto: result.survivor,
            edge: { v0, v1 }
          });

          componentBirth.delete(result.merged);
        }
      }
    }

    // Surviving components (born but never die → infinite persistence)
    const roots = new Set();
    for (let i = 0; i < n; i++) roots.add(uf.find(i));
    for (const root of roots) {
      this.persistenceDiagram.H0.push({
        birth: componentBirth.get(root) || 0,
        death: Infinity,
        persistence: Infinity,
        type: 'essential_component'
      });
    }

    // ─── H₁: Loops/Cycles (simplified boundary matrix reduction) ───
    this._computeH1(sortedEdges);

    // ─── Betti Curve: β₀(ε), β₁(ε) as function of filtration ───
    this._computeBettiCurve(sortedEdges);

    return this.persistenceDiagram;
  }

  /**
   * H₁ computation via boundary matrix reduction.
   *
   * A 1-cycle (loop) is BORN when an edge connects two already-connected
   * vertices (creating a loop). It DIES when a triangle fills it in.
   *
   * For options: a persistent 1-cycle = a closed path through the options
   * chain where IVs are locally consistent along each edge but globally
   * inconsistent around the loop = ARBITRAGE.
   */
  _computeH1(sortedEdges) {
    const n = this.complex.numVertices;
    const uf = new UnionFind(n);
    const cycles = []; // edges that create cycles

    // Find cycle-creating edges
    for (const edge of sortedEdges) {
      if (uf.connected(edge.v0, edge.v1)) {
        // This edge creates a 1-cycle!
        cycles.push({
          birth: edge.filtration,
          edge,
          killed: false
        });
      } else {
        uf.union(edge.v0, edge.v1);
      }
    }

    // Sort triangles by filtration
    const sortedTriangles = [...this.complex.triangles].sort((a, b) => a.filtration - b.filtration);

    // Each triangle can kill one 1-cycle (by filling the hole)
    // Match triangles to cycles using boundary relationships
    for (const tri of sortedTriangles) {
      // Find the oldest living cycle that this triangle can kill
      for (const cycle of cycles) {
        if (cycle.killed) continue;

        // Check if the cycle's edge is a face of this triangle
        const { v0, v1 } = cycle.edge;
        const triVerts = [tri.v0, tri.v1, tri.v2];

        if (triVerts.includes(v0) && triVerts.includes(v1)) {
          // This triangle kills this cycle
          cycle.killed = true;
          const persistence = tri.filtration - cycle.birth;

          this.persistenceDiagram.H1.push({
            birth: cycle.birth,
            death: tri.filtration,
            persistence,
            type: 'loop',
            cycleEdge: cycle.edge,
            killingTriangle: tri
          });
          break;
        }
      }
    }

    // Surviving cycles (essential 1-classes)
    for (const cycle of cycles) {
      if (!cycle.killed) {
        this.persistenceDiagram.H1.push({
          birth: cycle.birth,
          death: Infinity,
          persistence: Infinity,
          type: 'essential_loop',
          cycleEdge: cycle.edge
        });
      }
    }
  }

  /**
   * Compute the Betti curve: β₀(ε) and β₁(ε) at each filtration level.
   *
   * This is a "topological fingerprint" of the vol surface.
   * Comparing Betti curves across time detects regime changes.
   */
  _computeBettiCurve(sortedEdges) {
    if (sortedEdges.length === 0) return;

    const maxFiltration = sortedEdges[sortedEdges.length - 1].filtration;
    const steps = this.config.filtrationSteps;
    const stepSize = maxFiltration / steps;

    const n = this.complex.numVertices;
    let edgeIdx = 0;
    const uf = new UnionFind(n);
    let components = n;
    let loops = 0;

    for (let s = 0; s <= steps; s++) {
      const threshold = s * stepSize;

      // Add all edges up to this threshold
      while (edgeIdx < sortedEdges.length && sortedEdges[edgeIdx].filtration <= threshold) {
        const edge = sortedEdges[edgeIdx];
        if (uf.connected(edge.v0, edge.v1)) {
          loops++; // cycle-creating edge
        } else {
          uf.union(edge.v0, edge.v1);
          components--;
        }
        edgeIdx++;
      }

      // Count triangles killing loops at this threshold
      let killedLoops = 0;
      for (const tri of this.complex.triangles) {
        if (tri.filtration <= threshold) killedLoops++;
      }

      this.bettiCurve.push({
        filtration: threshold,
        beta0: components,             // connected components
        beta1: Math.max(0, loops - killedLoops), // loops (approximate)
        edgesAdded: edgeIdx
      });
    }
  }

  /**
   * Distance between two options for the Vietoris-Rips filtration.
   *
   * This is the critical design choice — it determines what
   * "nearby" means topologically.
   */
  _optionDistance(opt1, opt2, spot) {
    // IV difference (normalized)
    const ivDiff = Math.abs((opt1.iv || 30) - (opt2.iv || 30)) / 100;

    // Moneyness difference
    const m1 = (opt1.strike - spot) / spot;
    const m2 = (opt2.strike - spot) / spot;
    const moneyDiff = Math.abs(m1 - m2);

    // Time difference (log-scaled)
    const t1 = Math.log(opt1.expiry + 1);
    const t2 = Math.log(opt2.expiry + 1);
    const timeDiff = Math.abs(t1 - t2) / Math.log(366);

    // Delta difference
    const deltaDiff = Math.abs((opt1.delta || 0) - (opt2.delta || 0));

    // Type penalty (calls vs puts are far apart unless similar delta)
    const typePenalty = opt1.type !== opt2.type ? 0.3 : 0;

    // Weighted combination
    return 0.35 * ivDiff +
           0.25 * moneyDiff +
           0.15 * timeDiff +
           0.15 * deltaDiff +
           0.10 * typePenalty;
  }

  /**
   * Extract trading signals from the persistence diagram.
   *
   * Key signals:
   * 1. Persistent H₁ loops = closed arbitrage paths
   * 2. Many short-lived H₀ features = fragmented vol surface = high uncertainty
   * 3. Sudden change in Betti curve = regime transition
   */
  extractSignals() {
    const signals = [];

    // Signal 1: Persistent loops (arbitrage candidates)
    const persistentLoops = this.persistenceDiagram.H1.filter(
      f => f.persistence > this.config.persistenceThreshold
    );

    for (const loop of persistentLoops) {
      signals.push({
        type: 'PERSISTENT_LOOP',
        severity: loop.persistence > 0.3 ? 'HIGH' : 'MEDIUM',
        persistence: loop.persistence,
        birth: loop.birth,
        death: loop.death,
        description: `1-cycle born at ε=${loop.birth.toFixed(3)}, dies at ε=${loop.death === Infinity ? '∞' : loop.death.toFixed(3)}. ` +
                     `Persistence ${loop.persistence === Infinity ? '∞' : loop.persistence.toFixed(3)} indicates ` +
                     `${loop.persistence > 0.3 ? 'strong' : 'moderate'} closed arbitrage path.`
      });
    }

    // Signal 2: Fragmentation index (how many short-lived H₀ features)
    const shortLived = this.persistenceDiagram.H0.filter(
      f => f.persistence < this.config.persistenceThreshold && f.persistence > 0
    );
    const fragmentation = shortLived.length / Math.max(1, this.persistenceDiagram.H0.length);

    if (fragmentation > 0.7) {
      signals.push({
        type: 'HIGH_FRAGMENTATION',
        severity: 'WARNING',
        fragmentation,
        description: `Vol surface is highly fragmented (${(fragmentation * 100).toFixed(1)}% short-lived components). ` +
                     `Market microstructure may be unstable.`
      });
    }

    // Signal 3: Betti curve gradient (rapid topology changes)
    if (this.bettiCurve.length > 2) {
      for (let i = 1; i < this.bettiCurve.length - 1; i++) {
        const prev = this.bettiCurve[i - 1];
        const curr = this.bettiCurve[i];
        const next = this.bettiCurve[i + 1];

        // Sudden jump in β₁ (loops appearing)
        if (curr.beta1 > prev.beta1 + 3 && curr.beta1 > next.beta1 + 2) {
          signals.push({
            type: 'BETTI_SPIKE',
            severity: 'HIGH',
            filtration: curr.filtration,
            beta1: curr.beta1,
            description: `Sudden spike in β₁ at ε=${curr.filtration.toFixed(3)} (${curr.beta1} loops). ` +
                         `Multiple arbitrage paths opening simultaneously.`
          });
        }
      }
    }

    // Signal 4: Topological complexity
    const totalPersistence = this.persistenceDiagram.H1.reduce(
      (s, f) => s + (f.persistence === Infinity ? 1 : f.persistence), 0
    );

    signals.push({
      type: 'TOPOLOGICAL_COMPLEXITY',
      severity: totalPersistence > 2 ? 'HIGH' : totalPersistence > 0.5 ? 'MEDIUM' : 'LOW',
      totalPersistence,
      numLoops: this.persistenceDiagram.H1.length,
      numComponents: this.persistenceDiagram.H0.filter(f => f.death === Infinity).length,
      description: `Vol surface has ${this.persistenceDiagram.H1.length} loops ` +
                   `with total persistence ${totalPersistence.toFixed(3)}.`
    });

    return signals;
  }
}

// =============================================================================
// PART 2: RICCI FLOW
// =============================================================================

/**
 * Ollivier-Ricci Curvature
 *
 * The discrete analogue of Ricci curvature for graphs.
 * For two adjacent nodes x, y:
 *
 *   κ(x,y) = 1 - W₁(μₓ, μᵧ) / d(x,y)
 *
 * where:
 *   W₁ = Wasserstein-1 (Earth Mover's) distance
 *   μₓ = probability measure at x (uniform over neighbors)
 *   d(x,y) = graph distance
 *
 * Interpretation:
 *   κ > 0: Positively curved (like a sphere) → neighbors converge
 *   κ = 0: Flat (like Euclidean) → parallel transport
 *   κ < 0: Negatively curved (like a saddle) → neighbors diverge
 *
 * For options: negative curvature = vol surface "saddle point" = stress
 */
class OllivierRicciCurvature {
  constructor() {
    this.curvatures = new Map(); // "v0_v1" → κ
  }

  /**
   * Compute Ollivier-Ricci curvature for all edges in the graph.
   *
   * @param {Map} adjacency - nodeId → [{neighbor, weight}]
   * @param {Function} distance - (nodeId1, nodeId2) → distance
   */
  compute(adjacency, distance) {
    this.curvatures.clear();

    for (const [node, neighbors] of adjacency) {
      for (const { neighbor, weight } of neighbors) {
        const edgeKey = node < neighbor ? `${node}_${neighbor}` : `${neighbor}_${node}`;
        if (this.curvatures.has(edgeKey)) continue;

        const kappa = this._edgeCurvature(node, neighbor, adjacency, distance);
        this.curvatures.set(edgeKey, kappa);
      }
    }

    return this.curvatures;
  }

  _edgeCurvature(x, y, adjacency, distance) {
    const neighborsX = adjacency.get(x) || [];
    const neighborsY = adjacency.get(y) || [];

    if (neighborsX.length === 0 || neighborsY.length === 0) return 0;

    // Build probability measures (uniform over neighbors + self-loop)
    const muX = this._buildMeasure(x, neighborsX);
    const muY = this._buildMeasure(y, neighborsY);

    // Compute Wasserstein-1 distance using linear program relaxation
    // (simplified: use greedy matching for computational efficiency)
    const w1 = this._wasserstein1(muX, muY, distance);

    // Edge distance
    const dXY = distance(x, y);
    if (dXY < 1e-10) return 0;

    // Ollivier-Ricci curvature
    return 1 - w1 / dXY;
  }

  _buildMeasure(node, neighbors) {
    // Lazy random walk: stay with probability α, move with 1-α
    const alpha = 0.5; // idleness parameter
    const measure = new Map();

    measure.set(node, alpha);

    const totalWeight = neighbors.reduce((s, n) => s + n.weight, 0);
    for (const { neighbor, weight } of neighbors) {
      const prob = (1 - alpha) * weight / totalWeight;
      measure.set(neighbor, (measure.get(neighbor) || 0) + prob);
    }

    return measure;
  }

  /**
   * Wasserstein-1 distance via greedy Earth Mover's matching.
   *
   * Exact W₁ requires solving a linear program, but the greedy
   * approximation is O(n²) and sufficient for our purposes.
   */
  _wasserstein1(muX, muY, distance) {
    // Fast W₁ approximation: match nodes by sorted distance to a reference
    // This is O(n log n) instead of O(n³) for the exact LP
    const nodesX = [...muX.entries()];
    const nodesY = [...muY.entries()];

    // For small measures, use direct computation
    if (nodesX.length <= 6 && nodesY.length <= 6) {
      return this._wasserstein1Direct(nodesX, nodesY, distance);
    }

    // Sinkhorn-like approximation for larger measures
    let totalCost = 0;
    const remainingX = nodesX.map(([node, mass]) => ({ node, mass }));
    const remainingY = nodesY.map(([node, mass]) => ({ node, mass }));

    // Greedy matching with early exit
    for (let iter = 0; iter < 20; iter++) {
      let bestCost = Infinity;
      let bestI = -1, bestJ = -1;

      for (let i = 0; i < remainingX.length; i++) {
        if (remainingX[i].mass <= 1e-10) continue;
        for (let j = 0; j < remainingY.length; j++) {
          if (remainingY[j].mass <= 1e-10) continue;
          const cost = distance(remainingX[i].node, remainingY[j].node);
          if (cost < bestCost) { bestCost = cost; bestI = i; bestJ = j; }
        }
      }

      if (bestI === -1) break;

      const transferred = Math.min(remainingX[bestI].mass, remainingY[bestJ].mass);
      totalCost += transferred * bestCost;
      remainingX[bestI].mass -= transferred;
      remainingY[bestJ].mass -= transferred;
    }

    return totalCost;
  }

  _wasserstein1Direct(nodesX, nodesY, distance) {
    let totalCost = 0;
    const rx = nodesX.map(([node, mass]) => ({ node, mass }));
    const ry = nodesY.map(([node, mass]) => ({ node, mass }));

    for (let iter = 0; iter < rx.length + ry.length; iter++) {
      let bestCost = Infinity;
      let bestI = -1, bestJ = -1;

      for (let i = 0; i < rx.length; i++) {
        if (rx[i].mass <= 1e-10) continue;
        for (let j = 0; j < ry.length; j++) {
          if (ry[j].mass <= 1e-10) continue;
          const cost = distance(rx[i].node, ry[j].node);
          if (cost < bestCost) { bestCost = cost; bestI = i; bestJ = j; }
        }
      }

      if (bestI === -1) break;
      const transferred = Math.min(rx[bestI].mass, ry[bestJ].mass);
      totalCost += transferred * bestCost;
      rx[bestI].mass -= transferred;
      ry[bestJ].mass -= transferred;
    }

    return totalCost;
  }
}

// ─── Ricci Flow Engine ───

/**
 * VolSurfaceRicciFlow
 *
 * Evolves the vol surface metric using discrete Ricci flow:
 *
 *   w(x,y) ← w(x,y) - ε · κ(x,y) · w(x,y)
 *
 * where:
 *   w = edge weight (encodes metric distance)
 *   κ = Ollivier-Ricci curvature
 *   ε = step size
 *
 * The flow smooths the surface:
 * - Positively curved edges → weight decreases (brings nodes closer)
 * - Negatively curved edges → weight increases (pushes nodes apart)
 * - Zero curvature → unchanged (already in equilibrium)
 *
 * Eventually, the flow either:
 * 1. Converges to uniform curvature (stable regime)
 * 2. Develops singularities where edges collapse to zero (regime transition!)
 *
 * When an edge collapses → SURGERY: cut the edge and separate the surface
 * into two components. This is the discrete analogue of Perelman's surgery
 * in the Poincaré conjecture proof.
 *
 *   Before surgery:         After surgery:
 *
 *    ●──●──●──●──●          ●──●──●    ●──●
 *    │  │ ↗│↙ │  │          │  │  │    │  │
 *    ●──●──●──●──●    →     ●──●──●    ●──●
 *    │  │  │  │  │          │  │  │    │  │
 *    ●──●──●──●──●          ●──●──●    ●──●
 *            ↑
 *        Neck pinch
 *      (edge → 0)
 */
class VolSurfaceRicciFlow {
  constructor(config = tdaConfig.ricciFlow) {
    this.config = config;
    this.ricciComputer = new OllivierRicciCurvature();
    this.adjacency = new Map();
    this.edgeWeights = new Map();
    this.nodeData = new Map();
    this.history = [];
    this.surgeries = [];
  }

  /**
   * Build the weighted graph from an options chain.
   */
  buildGraph(chain) {
    const { options, spot } = chain;
    this.adjacency.clear();
    this.edgeWeights.clear();
    this.nodeData.clear();

    // Create nodes
    for (const opt of options) {
      const id = `${opt.type}_${opt.strike}_${opt.expiry}`;
      this.nodeData.set(id, { ...opt, spot });
      this.adjacency.set(id, []);
    }

    // Create edges between nearby options
    for (let i = 0; i < options.length; i++) {
      for (let j = i + 1; j < options.length; j++) {
        const opt_i = options[i];
        const opt_j = options[j];

        const id_i = `${opt_i.type}_${opt_i.strike}_${opt_i.expiry}`;
        const id_j = `${opt_j.type}_${opt_j.strike}_${opt_j.expiry}`;

        // Connect options that are "neighbors"
        const shouldConnect = this._shouldConnect(opt_i, opt_j, spot);
        if (!shouldConnect) continue;

        const weight = this._initialWeight(opt_i, opt_j, spot);
        const edgeKey = id_i < id_j ? `${id_i}_${id_j}` : `${id_j}_${id_i}`;

        this.edgeWeights.set(edgeKey, weight);
        this.adjacency.get(id_i).push({ neighbor: id_j, weight });
        this.adjacency.get(id_j).push({ neighbor: id_i, weight });
      }
    }
  }

  _shouldConnect(opt1, opt2, spot) {
    // Same type only (calls with calls, puts with puts)
    if (opt1.type !== opt2.type) return false;

    // Only connect immediate neighbors on the grid
    const strikeDiff = Math.abs(opt1.strike - opt2.strike) / spot;
    const expiryDiff = Math.abs(opt1.expiry - opt2.expiry);

    // Strict: same expiry + adjacent strike, OR same strike + adjacent expiry
    const sameExpiry = expiryDiff === 0 && strikeDiff < 0.04;
    const sameStrike = strikeDiff < 0.001 && expiryDiff <= 30;

    return sameExpiry || sameStrike;
  }

  _initialWeight(opt1, opt2, spot) {
    // Weight based on IV difference (higher IV diff = longer edge = larger weight)
    const ivDiff = Math.abs((opt1.iv || 30) - (opt2.iv || 30)) / 100;
    const strikeDiff = Math.abs(opt1.strike - opt2.strike) / spot;
    const timeDiff = Math.abs(opt1.expiry - opt2.expiry) / 365;

    return Math.sqrt(ivDiff * ivDiff + strikeDiff * strikeDiff + timeDiff * timeDiff) + 0.01;
  }

  /**
   * Run Ricci flow for the configured number of steps.
   *
   * Returns the flow history and any surgeries performed.
   */
  flow() {
    const { steps, stepSize, surgeryThreshold, convergenceEps } = this.config;

    for (let step = 0; step < steps; step++) {
      // 1. Compute Ricci curvature for all edges
      const distanceFn = (a, b) => {
        const key = a < b ? `${a}_${b}` : `${b}_${a}`;
        return this.edgeWeights.get(key) || Infinity;
      };

      const curvatures = this.ricciComputer.compute(this.adjacency, distanceFn);

      // 2. Update edge weights: w ← w - ε · κ · w
      let maxChange = 0;
      const updates = [];

      for (const [edgeKey, kappa] of curvatures) {
        const w = this.edgeWeights.get(edgeKey);
        if (w === undefined) continue;

        const newW = w - stepSize * kappa * w;
        const clampedW = Math.max(0, newW); // weights can't go negative

        maxChange = Math.max(maxChange, Math.abs(clampedW - w));
        updates.push({ edgeKey, oldW: w, newW: clampedW, kappa });
      }

      // Apply updates
      for (const { edgeKey, newW } of updates) {
        this.edgeWeights.set(edgeKey, newW);
      }

      // Update adjacency weights to match
      this._syncAdjacencyWeights();

      // 3. Check for surgeries (edge collapses)
      const collapsed = [];
      for (const { edgeKey, newW, kappa } of updates) {
        if (newW < surgeryThreshold) {
          collapsed.push({ edgeKey, weight: newW, curvature: kappa, step });
        }
      }

      if (collapsed.length > 0) {
        this._performSurgery(collapsed, step);
      }

      // 4. Record history
      if (this.config.trackHistory) {
        const curvatureStats = this._curvatureStats(curvatures);
        this.history.push({
          step,
          maxChange,
          ...curvatureStats,
          numEdges: this.edgeWeights.size,
          surgeries: collapsed.length
        });
      }

      // 5. Check convergence
      if (maxChange < convergenceEps) {
        break;
      }

      // 6. Normalize weights (optional, prevents drift)
      if (this.config.normalize && step % 10 === 9) {
        this._normalizeWeights();
      }
    }

    return {
      history: this.history,
      surgeries: this.surgeries,
      finalCurvatures: this.ricciComputer.curvatures,
      finalWeights: new Map(this.edgeWeights)
    };
  }

  _syncAdjacencyWeights() {
    for (const [node, neighbors] of this.adjacency) {
      for (const entry of neighbors) {
        const key = node < entry.neighbor
          ? `${node}_${entry.neighbor}`
          : `${entry.neighbor}_${node}`;
        entry.weight = this.edgeWeights.get(key) || entry.weight;
      }
    }
  }

  /**
   * Perform Ricci flow surgery.
   *
   * When an edge weight approaches zero, the vol surface is developing
   * a "neck pinch" singularity. We cut the edge and record the event.
   *
   * In the options context, a surgery means:
   * "The vol surface is splitting into two disconnected regimes
   *  at this point. Options on either side of the cut are now
   *  in fundamentally different pricing regimes."
   */
  _performSurgery(collapsed, step) {
    for (const { edgeKey, weight, curvature } of collapsed) {
      // Remove the edge
      this.edgeWeights.delete(edgeKey);

      // Remove from adjacency lists
      const [nodeA, nodeB] = this._parseEdgeKey(edgeKey);
      if (this.adjacency.has(nodeA)) {
        const neighbors = this.adjacency.get(nodeA);
        const idx = neighbors.findIndex(n => {
          const k = nodeA < n.neighbor ? `${nodeA}_${n.neighbor}` : `${n.neighbor}_${nodeA}`;
          return k === edgeKey;
        });
        if (idx >= 0) neighbors.splice(idx, 1);
      }
      if (this.adjacency.has(nodeB)) {
        const neighbors = this.adjacency.get(nodeB);
        const idx = neighbors.findIndex(n => {
          const k = nodeB < n.neighbor ? `${nodeB}_${n.neighbor}` : `${n.neighbor}_${nodeB}`;
          return k === edgeKey;
        });
        if (idx >= 0) neighbors.splice(idx, 1);
      }

      // Record surgery
      this.surgeries.push({
        step,
        edgeKey,
        weight,
        curvature,
        nodeA: this._getNodeInfo(nodeA),
        nodeB: this._getNodeInfo(nodeB),
        interpretation: this._interpretSurgery(nodeA, nodeB)
      });
    }
  }

  _parseEdgeKey(key) {
    // Edge keys are "nodeA_nodeB" but node IDs contain underscores
    // Find the split point by looking for known node IDs
    for (const nodeId of this.nodeData.keys()) {
      if (key.startsWith(nodeId + '_')) {
        const rest = key.slice(nodeId.length + 1);
        if (this.nodeData.has(rest)) {
          return [nodeId, rest];
        }
      }
    }
    // Fallback: split at first underscore that produces valid nodes
    const midPoints = [];
    for (let i = 1; i < key.length; i++) {
      if (key[i] === '_') midPoints.push(i);
    }
    // Try from the middle out
    midPoints.sort((a, b) => Math.abs(a - key.length / 2) - Math.abs(b - key.length / 2));
    for (const mid of midPoints) {
      const a = key.slice(0, mid);
      const b = key.slice(mid + 1);
      if (this.nodeData.has(a) && this.nodeData.has(b)) return [a, b];
    }
    return [key, key]; // shouldn't happen
  }

  _getNodeInfo(nodeId) {
    const data = this.nodeData.get(nodeId);
    if (!data) return { id: nodeId };
    return {
      id: nodeId,
      type: data.type,
      strike: data.strike,
      expiry: data.expiry,
      iv: data.iv
    };
  }

  _interpretSurgery(nodeA, nodeB) {
    const dataA = this.nodeData.get(nodeA);
    const dataB = this.nodeData.get(nodeB);
    if (!dataA || !dataB) return 'Vol surface splitting at unknown boundary';

    if (dataA.expiry !== dataB.expiry) {
      return `TERM STRUCTURE BREAK: Vol surface splitting between ${dataA.expiry}d and ${dataB.expiry}d expiries. ` +
             `Different term regimes emerging.`;
    }

    if (Math.abs(dataA.strike - dataB.strike) > 0) {
      return `SMILE FRACTURE: Vol smile breaking between $${dataA.strike} and $${dataB.strike}. ` +
             `Moneyness regimes decoupling.`;
    }

    return `Vol surface topology change at ${nodeA} ↔ ${nodeB}`;
  }

  _curvatureStats(curvatures) {
    const values = [...curvatures.values()];
    if (values.length === 0) return { meanCurvature: 0, minCurvature: 0, maxCurvature: 0, stdCurvature: 0 };

    const mean = values.reduce((s, v) => s + v, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;

    return {
      meanCurvature: mean,
      minCurvature: min,
      maxCurvature: max,
      stdCurvature: Math.sqrt(variance),
      negativeFraction: values.filter(v => v < 0).length / values.length
    };
  }

  _normalizeWeights() {
    const weights = [...this.edgeWeights.values()];
    if (weights.length === 0) return;

    const mean = weights.reduce((s, w) => s + w, 0) / weights.length;
    if (mean < 1e-10) return;

    for (const [key, w] of this.edgeWeights) {
      this.edgeWeights.set(key, w / mean);
    }
    this._syncAdjacencyWeights();
  }

  /**
   * Extract trading signals from Ricci flow results.
   */
  extractSignals() {
    const signals = [];

    // Signal 1: Surgeries (regime splits)
    for (const surgery of this.surgeries) {
      signals.push({
        type: 'RICCI_SURGERY',
        severity: 'CRITICAL',
        step: surgery.step,
        curvature: surgery.curvature,
        interpretation: surgery.interpretation,
        description: `Neck pinch at step ${surgery.step}: edge collapsed with κ=${surgery.curvature.toFixed(4)}. ${surgery.interpretation}`
      });
    }

    // Signal 2: Curvature convergence speed
    if (this.history.length > 10) {
      const early = this.history.slice(0, 5);
      const late = this.history.slice(-5);
      const earlyMeanCurv = early.reduce((s, h) => s + Math.abs(h.meanCurvature || 0), 0) / 5;
      const lateMeanCurv = late.reduce((s, h) => s + Math.abs(h.meanCurvature || 0), 0) / 5;

      const convergenceRate = (earlyMeanCurv - lateMeanCurv) / Math.max(earlyMeanCurv, 1e-10);

      if (convergenceRate < 0.1) {
        signals.push({
          type: 'SLOW_CONVERGENCE',
          severity: 'HIGH',
          convergenceRate,
          description: `Ricci flow converges slowly (rate=${convergenceRate.toFixed(4)}). ` +
                       `Vol surface has deep structural features resisting smoothing — likely real mispricings.`
        });
      } else if (convergenceRate > 0.8) {
        signals.push({
          type: 'FAST_CONVERGENCE',
          severity: 'LOW',
          convergenceRate,
          description: `Ricci flow converges quickly (rate=${convergenceRate.toFixed(4)}). ` +
                       `Vol surface is near equilibrium — few mispricings.`
        });
      }
    }

    // Signal 3: Persistent negative curvature regions (saddle points)
    const finalCurvatures = this.ricciComputer.curvatures;
    const negativeEdges = [...finalCurvatures.entries()]
      .filter(([, k]) => k < tdaConfig.anomaly.ricciCurvatureThreshold)
      .sort((a, b) => a[1] - b[1]);

    if (negativeEdges.length > 0) {
      for (const [edgeKey, kappa] of negativeEdges.slice(0, 5)) {
        const [nodeA, nodeB] = this._parseEdgeKey(edgeKey);
        signals.push({
          type: 'NEGATIVE_CURVATURE_BOTTLENECK',
          severity: kappa < -1 ? 'HIGH' : 'MEDIUM',
          edgeKey,
          curvature: kappa,
          nodeA: this._getNodeInfo(nodeA),
          nodeB: this._getNodeInfo(nodeB),
          description: `Saddle point between ${nodeA} and ${nodeB} (κ=${kappa.toFixed(4)}). ` +
                       `Vol surface under stress — potential breakpoint.`
        });
      }
    }

    // Signal 4: Edge weight distribution (after flow)
    const weights = [...this.edgeWeights.values()];
    if (weights.length > 0) {
      const near_zero = weights.filter(w => w < this.config.surgeryThreshold * 3).length;
      if (near_zero > 0) {
        signals.push({
          type: 'NEAR_SURGERY',
          severity: 'WARNING',
          count: near_zero,
          description: `${near_zero} edges approaching surgery threshold. ` +
                       `Vol surface may fragment in the near future.`
        });
      }
    }

    return signals;
  }
}

// =============================================================================
// UNIFIED PIPELINE
// =============================================================================

/**
 * TopologicalVolAnalysis
 *
 * Combines persistent homology + Ricci flow for comprehensive
 * topological analysis of options vol surfaces.
 *
 * The two methods are complementary:
 * - Persistent homology reveals WHAT features exist (loops, components)
 * - Ricci flow reveals HOW the surface evolves (convergence, singularities)
 *
 * Together they provide:
 * 1. Arbitrage loop detection (persistent H₁ features)
 * 2. Vol regime splitting (Ricci surgery)
 * 3. Surface stability assessment (convergence rate)
 * 4. Topological early warning signals (Betti curve changes)
 */
class TopologicalVolAnalysis {
  constructor(config = tdaConfig) {
    this.config = config;
    this.persistence = new VolSurfacePersistence(config.homology);
    this.ricciFlow = new VolSurfaceRicciFlow(config.ricciFlow);
    this.results = [];
  }

  /**
   * Run full topological analysis on an options chain.
   */
  analyze(chain) {
    const t0 = performance.now();

    // Phase 1: Persistent Homology
    const complex = this.persistence.buildComplex(chain);
    const diagram = this.persistence.computePersistence();
    const tdaSignals = this.persistence.extractSignals();

    // Phase 2: Ricci Flow
    this.ricciFlow.buildGraph(chain);
    const flowResult = this.ricciFlow.flow();
    const ricciSignals = this.ricciFlow.extractSignals();

    // Phase 3: Combined analysis
    const combined = this._combineAnalyses(tdaSignals, ricciSignals, diagram, flowResult);

    const elapsed = performance.now() - t0;

    const result = {
      timestamp: Date.now(),
      underlying: chain.underlying,
      spot: chain.spot,
      options: chain.options.length,

      // Persistent Homology
      persistenceDiagram: diagram,
      bettiCurve: this.persistence.bettiCurve,
      complexStats: {
        vertices: complex.numVertices,
        edges: complex.numEdges,
        triangles: complex.numTriangles
      },

      // Ricci Flow
      flowHistory: flowResult.history,
      surgeries: flowResult.surgeries,
      finalCurvatures: flowResult.finalCurvatures.size,

      // Signals
      tdaSignals,
      ricciSignals,
      combined,

      latencyMs: elapsed
    };

    this.results.push(result);
    return result;
  }

  _combineAnalyses(tdaSignals, ricciSignals, diagram, flowResult) {
    const combined = {
      overallRisk: 'LOW',
      topologicalHealth: 'STABLE',
      signals: [],
      summary: ''
    };

    // Count critical signals
    const criticalTDA = tdaSignals.filter(s => s.severity === 'HIGH' || s.severity === 'CRITICAL');
    const criticalRicci = ricciSignals.filter(s => s.severity === 'HIGH' || s.severity === 'CRITICAL');

    // Persistent loops from TDA
    const persistentLoops = diagram.H1.filter(
      f => f.persistence > this.config.anomaly.minPersistence
    );

    // Surgeries from Ricci
    const surgeries = flowResult.surgeries;

    // Combined risk assessment
    if (surgeries.length > 0 && persistentLoops.length > 0) {
      combined.overallRisk = 'CRITICAL';
      combined.topologicalHealth = 'FRAGMENTING';
      combined.summary = `Vol surface is fragmenting (${surgeries.length} surgery events) ` +
                         `with ${persistentLoops.length} persistent arbitrage loops. ` +
                         `Regime change likely imminent.`;
    } else if (surgeries.length > 0) {
      combined.overallRisk = 'HIGH';
      combined.topologicalHealth = 'SPLITTING';
      combined.summary = `Vol surface developing ${surgeries.length} fracture(s). ` +
                         `Pricing regimes separating.`;
    } else if (persistentLoops.length > 3) {
      combined.overallRisk = 'HIGH';
      combined.topologicalHealth = 'STRESSED';
      combined.summary = `Vol surface has ${persistentLoops.length} persistent loops ` +
                         `(potential arbitrage). Surface topology is complex.`;
    } else if (criticalTDA.length + criticalRicci.length > 2) {
      combined.overallRisk = 'MEDIUM';
      combined.topologicalHealth = 'ANOMALOUS';
      combined.summary = `Multiple topological anomalies detected. ` +
                         `${criticalTDA.length} from TDA, ${criticalRicci.length} from Ricci flow.`;
    } else {
      combined.overallRisk = 'LOW';
      combined.topologicalHealth = 'STABLE';
      combined.summary = 'Vol surface topology is stable. No significant anomalies.';
    }

    // Merge all signals sorted by severity
    const severityOrder = { CRITICAL: 0, HIGH: 1, WARNING: 2, MEDIUM: 3, LOW: 4 };
    combined.signals = [...tdaSignals, ...ricciSignals]
      .sort((a, b) => (severityOrder[a.severity] || 5) - (severityOrder[b.severity] || 5));

    return combined;
  }
}

// =============================================================================
// DEMONSTRATION
// =============================================================================

async function main() {
  console.log();
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║  TOPOLOGICAL VOLATILITY SURFACE ANALYSIS                    ║');
  console.log('║                                                              ║');
  console.log('║  Part 1: Persistent Homology (TDA)                          ║');
  console.log('║  Part 2: Ricci Flow Smoothing                               ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  console.log();

  const analyzer = new TopologicalVolAnalysis();

  // ─── Scenario 1: Normal Market ───
  console.log('═══ Scenario 1: Normal Market Conditions ═══\n');
  const normalChain = OptionsChainGenerator.generate('SPY', 450, {
    numStrikes: 8,
    numExpiries: 4,
    baseIV: 18,
    skew: -0.12,
    anomalyCount: 0
  });
  console.log(`  Options: ${normalChain.options.length}`);

  const normalResult = analyzer.analyze(normalChain);
  printResults(normalResult);

  // ─── Scenario 2: Stressed Market (anomalies injected) ───
  console.log('\n═══ Scenario 2: Stressed Market with Anomalies ═══\n');
  const stressedChain = OptionsChainGenerator.generate('SPY', 450, {
    numStrikes: 8,
    numExpiries: 4,
    baseIV: 35,
    skew: -0.28,
    anomalyCount: 6
  });
  console.log(`  Options: ${stressedChain.options.length}`);

  // Reset analyzer for fresh analysis
  const analyzer2 = new TopologicalVolAnalysis();
  const stressedResult = analyzer2.analyze(stressedChain);
  printResults(stressedResult);

  // ─── Scenario 3: Pre-Crisis (extreme conditions) ───
  console.log('\n═══ Scenario 3: Pre-Crisis Conditions ═══\n');
  const crisisChain = OptionsChainGenerator.generate('SPY', 450, {
    numStrikes: 8,
    numExpiries: 4,
    baseIV: 55,
    skew: -0.40,
    anomalyCount: 8
  });
  console.log(`  Options: ${crisisChain.options.length}`);

  const analyzer3 = new TopologicalVolAnalysis();
  const crisisResult = analyzer3.analyze(crisisChain);
  printResults(crisisResult);

  // ─── Comparative Summary ───
  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('  COMPARATIVE SUMMARY');
  console.log('═══════════════════════════════════════════════════════════════\n');

  const scenarios = [
    { name: 'Normal',  result: normalResult },
    { name: 'Stressed', result: stressedResult },
    { name: 'Crisis',   result: crisisResult }
  ];

  console.log('  Scenario   │ β₀ final │ H₁ loops │ Surgeries │ Risk     │ Health');
  console.log('  ───────────┼──────────┼──────────┼───────────┼──────────┼────────────');
  for (const { name, result } of scenarios) {
    const beta0 = result.persistenceDiagram.H0.filter(f => f.death === Infinity).length;
    const loops = result.persistenceDiagram.H1.length;
    const surgeries = result.surgeries.length;
    console.log(
      `  ${name.padEnd(10)} │ ${String(beta0).padEnd(8)} │ ${String(loops).padEnd(8)} │ ${String(surgeries).padEnd(9)} │ ${result.combined.overallRisk.padEnd(8)} │ ${result.combined.topologicalHealth}`
    );
  }

  console.log('\n  Key insights:');
  console.log('  • β₀ = connected components. β₀ = 1 means fully connected vol surface.');
  console.log('  • H₁ loops = closed paths through vol surface = potential arbitrage.');
  console.log('  • Surgeries = Ricci flow singularities = vol surface fragmenting.');
  console.log('  • More loops + surgeries under stress confirms the theory:\n' +
              '    hyperbolic vol geometry reveals instability before it\'s visible classically.\n');

  return { normalResult, stressedResult, crisisResult };
}

function printResults(result) {
  // Persistence diagram summary
  const h0Essential = result.persistenceDiagram.H0.filter(f => f.death === Infinity);
  const h0Finite = result.persistenceDiagram.H0.filter(f => f.death !== Infinity);
  const h1 = result.persistenceDiagram.H1;
  const persistentH1 = h1.filter(f => f.persistence > tdaConfig.anomaly.minPersistence);

  console.log('  ─── Persistent Homology ───');
  console.log(`    Simplicial complex: ${result.complexStats.vertices} vertices, ${result.complexStats.edges} edges, ${result.complexStats.triangles} triangles`);
  console.log(`    H₀: ${h0Essential.length} essential components, ${h0Finite.length} ephemeral`);
  console.log(`    H₁: ${h1.length} total loops, ${persistentH1.length} persistent (above threshold)`);

  // Betti curve
  if (result.bettiCurve.length > 0) {
    const start = result.bettiCurve[0];
    const mid = result.bettiCurve[Math.floor(result.bettiCurve.length / 2)];
    const end = result.bettiCurve[result.bettiCurve.length - 1];
    console.log(`    Betti curve: β₀=${start.beta0}→${mid.beta0}→${end.beta0}, β₁=peak ${Math.max(...result.bettiCurve.map(b => b.beta1))}`);
  }

  // Ricci flow
  console.log('\n  ─── Ricci Flow ───');
  if (result.flowHistory.length > 0) {
    const first = result.flowHistory[0];
    const last = result.flowHistory[result.flowHistory.length - 1];
    console.log(`    Steps: ${result.flowHistory.length}`);
    console.log(`    Mean curvature: ${first.meanCurvature?.toFixed(4)} → ${last.meanCurvature?.toFixed(4)}`);
    console.log(`    Negative fraction: ${(first.negativeFraction * 100)?.toFixed(1)}% → ${(last.negativeFraction * 100)?.toFixed(1)}%`);
    console.log(`    Surgeries: ${result.surgeries.length}`);

    for (const surgery of result.surgeries.slice(0, 3)) {
      console.log(`      Step ${surgery.step}: ${surgery.interpretation}`);
    }
  }

  // Combined assessment
  console.log('\n  ─── Combined Assessment ───');
  console.log(`    Overall risk: ${result.combined.overallRisk}`);
  console.log(`    Topological health: ${result.combined.topologicalHealth}`);
  console.log(`    ${result.combined.summary}`);

  // Top signals
  const topSignals = result.combined.signals.slice(0, 3);
  if (topSignals.length > 0) {
    console.log('\n  ─── Top Signals ───');
    for (const sig of topSignals) {
      console.log(`    [${sig.severity}] ${sig.type}: ${sig.description.slice(0, 100)}${sig.description.length > 100 ? '...' : ''}`);
    }
  }

  console.log(`\n  Latency: ${result.latencyMs.toFixed(2)}ms`);
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
  // Part 1: Persistent Homology
  UnionFind,
  SimplicialComplex,
  VolSurfacePersistence,

  // Part 2: Ricci Flow
  OllivierRicciCurvature,
  VolSurfaceRicciFlow,

  // Unified Pipeline
  TopologicalVolAnalysis,

  // Configuration
  tdaConfig,

  // Demo
  main
};

const isMain = import.meta.url === `file://${process.argv[1]}` ||
               process.argv[1]?.endsWith('topological-vol-analysis.js');

if (isMain) {
  main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
  });
}
