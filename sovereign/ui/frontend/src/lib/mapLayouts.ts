// Hand-authored node positions for each map. Coordinates are in arbitrary units
// (Cytoscape `preset` layout) — Cytoscape fits/centers them inside the viewport.
// Designed maps deserve designed layouts; auto-layout (cose) jittered between mounts
// and produced crossings.

export type Pos = { x: number; y: number };

export const MAP_LAYOUTS: Record<string, Record<number, Pos>> = {
  rulebook9: {
    0: { x: 60, y: 220 },   // I
    1: { x: 540, y: 220 },  // D
    2: { x: 300, y: 40 },   // N
    3: { x: 170, y: 250 },  // C1
    4: { x: 220, y: 340 },  // C2
    5: { x: 260, y: 140 },  // C3
    6: { x: 340, y: 140 },  // C4
    7: { x: 380, y: 340 },  // C5
    8: { x: 430, y: 250 },  // C6
  },
  default9: {
    0: { x: 60, y: 100 },   // Capital-I
    1: { x: 60, y: 300 },   // Industrial-I
    2: { x: 180, y: 200 },  // Border-I
    3: { x: 420, y: 200 },  // Border-D
    4: { x: 540, y: 300 },  // Industrial-D
    5: { x: 540, y: 100 },  // Capital-D
    6: { x: 300, y: 300 },  // Steppe
    7: { x: 300, y: 200 },  // Coast
    8: { x: 300, y: 80 },   // Highlands
  },
  frontier12: {
    0: { x: 60, y: 90 },    // I-Cap
    1: { x: 60, y: 200 },   // I-Ind1
    2: { x: 60, y: 310 },   // I-Ind2
    3: { x: 180, y: 200 },  // I-Border
    4: { x: 460, y: 200 },  // D-Border
    5: { x: 580, y: 90 },   // D-Ind
    6: { x: 580, y: 310 },  // D-Cap
    7: { x: 240, y: 310 },  // Marsh
    8: { x: 320, y: 220 },  // Forest
    9: { x: 400, y: 310 },  // Plain
    10: { x: 380, y: 130 }, // Mtn
    11: { x: 240, y: 130 }, // Coast
  },
};
