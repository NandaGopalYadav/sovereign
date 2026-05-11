import { useEffect, useRef } from "react";
import cytoscape, { Core, ElementDefinition } from "cytoscape";
import type { LiveState, MapInfo } from "../lib/api";
import { MAP_LAYOUTS } from "../lib/mapLayouts";

type Props = {
  map: MapInfo;
  state: LiveState | null;
};

const NATION_COLOR: Record<number, string> = {
  0: "#c95757", // invader
  1: "#5b8eda", // defender
  2: "#a0a4b0", // neutral
  3: "#d39459", // contested (warm amber)
};

const HOME_GLYPH: Record<number, string> = {
  0: "◆", // invader home
  1: "◆", // defender home
  2: "◆", // neutral home
};

// Mix two hex colors at 50/50 — used for edges spanning two different controllers.
function mix(a: string, b: string): string {
  const pa = parseInt(a.slice(1), 16);
  const pb = parseInt(b.slice(1), 16);
  const r = Math.round(((pa >> 16) + (pb >> 16)) / 2);
  const g = Math.round((((pa >> 8) & 0xff) + ((pb >> 8) & 0xff)) / 2);
  const bl = Math.round(((pa & 0xff) + (pb & 0xff)) / 2);
  return `#${((r << 16) | (g << 8) | bl).toString(16).padStart(6, "0")}`;
}

export default function MapGraph({ map, state }: Props) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!mountRef.current) return;
    const layout = MAP_LAYOUTS[map.name];

    const elements: ElementDefinition[] = [
      ...map.territories.map((t) => {
        const pos = layout?.[t.id];
        return {
          group: "nodes" as const,
          data: {
            id: `t${t.id}`,
            tid: t.id,
            name: t.name,
            home_of: t.home_of,
            strategic: t.strategic_value,
            label: t.name,
            // Pie wedge sizes (0..100) — updated each tick from live state.
            pieI: 0,
            pieD: 0,
            pieN: 0,
            ctlColor: NATION_COLOR[t.home_of] ?? NATION_COLOR[3],
            glow: 0,
          },
          ...(pos ? { position: { x: pos.x, y: pos.y } } : {}),
        };
      }),
      ...map.edges.map(([u, v]) => ({
        group: "edges" as const,
        data: {
          id: `e${u}-${v}`,
          source: `t${u}`,
          target: `t${v}`,
          edgeColor: "#2f323b",
          edgeWidth: 1.4,
        },
      })),
    ];

    const cy = cytoscape({
      container: mountRef.current,
      elements,
      layout: layout
        ? { name: "preset", padding: 36, fit: true }
        : { name: "cose", animate: false, idealEdgeLength: () => 90, padding: 24 },
      style: [
        {
          selector: "node",
          style: {
            shape: "round-hexagon",
            label: "data(label)",
            "background-color": "#11131a",
            "background-opacity": 1,
            "border-width": 2,
            "border-color": "data(ctlColor)",
            "border-opacity": 0.95,
            "text-valign": "bottom",
            "text-halign": "center",
            "text-margin-y": 8,
            "font-family": "Inter, system-ui, sans-serif",
            "font-size": 11,
            "font-weight": 600,
            color: "#e6e7ea",
            "text-outline-color": "#0b0d12",
            "text-outline-width": 2,
            width: 60,
            height: 60,
            "pie-size": "94%",
            "pie-1-background-color": NATION_COLOR[0],
            "pie-1-background-size": "data(pieI)" as unknown as number,
            "pie-1-background-opacity": 0.85,
            "pie-2-background-color": NATION_COLOR[1],
            "pie-2-background-size": "data(pieD)" as unknown as number,
            "pie-2-background-opacity": 0.85,
            "pie-3-background-color": NATION_COLOR[2],
            "pie-3-background-size": "data(pieN)" as unknown as number,
            "pie-3-background-opacity": 0.7,
            "overlay-color": "data(ctlColor)",
            "overlay-padding": 6,
            "overlay-opacity": "data(glow)" as unknown as number,
            "transition-property":
              "border-color background-color overlay-opacity width height",
            "transition-duration": 220,
          },
        },
        {
          selector: "node.contested",
          style: {
            "border-style": "dashed",
          },
        },
        {
          selector: "node.last-target",
          style: {
            "border-width": 3,
            "overlay-opacity": 0.18,
          },
        },
        {
          selector: "edge",
          style: {
            "line-color": "data(edgeColor)",
            width: "data(edgeWidth)" as unknown as number,
            "curve-style": "unbundled-bezier",
            "control-point-distances": [12],
            "control-point-weights": [0.5],
            opacity: 0.85,
            "transition-property": "line-color width",
            "transition-duration": 220,
          },
        },
      ],
      wheelSensitivity: 0.2,
      minZoom: 0.5,
      maxZoom: 2.5,
    });

    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [map.name]);

  // Reflect controller / unit-count changes whenever state updates.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !state) return;

    const controllerColors: Record<number, string> = {};
    const totals: Record<number, number> = {};

    cy.batch(() => {
      map.territories.forEach((t) => {
        const ctl = state.controller[t.id];
        const inv = state.invader_units[t.id] ?? 0;
        const def = state.defender_units[t.id] ?? 0;
        const neu = state.neutral_units[t.id] ?? 0;
        const total = inv + def + neu;
        totals[t.id] = total;
        const ctlColor =
          ctl === 3 ? NATION_COLOR[3] : NATION_COLOR[ctl] ?? NATION_COLOR[3];
        controllerColors[t.id] = ctlColor;

        const node = cy.getElementById(`t${t.id}`);
        if (!node || node.empty()) return;

        // Pie wedges show unit composition. If empty, render a faint controller-tinted disk.
        if (total > 0) {
          node.data("pieI", (inv / total) * 100);
          node.data("pieD", (def / total) * 100);
          node.data("pieN", (neu / total) * 100);
        } else {
          node.data("pieI", 0);
          node.data("pieD", 0);
          node.data("pieN", 0);
        }
        node.data("ctlColor", ctlColor);

        // Label: name on top line, strategic glyph for home territories, total units below.
        const homeMark =
          t.home_of < 3 && HOME_GLYPH[t.home_of] ? `${HOME_GLYPH[t.home_of]} ` : "";
        node.data(
          "label",
          total > 0
            ? `${homeMark}${t.name}\n${total.toFixed(0)}u`
            : `${homeMark}${t.name}`
        );

        // Slight size scaling with troop concentration, but capped to keep layout stable.
        const size = 56 + Math.min(total * 1.6, 22);
        node.style("width", size);
        node.style("height", size);

        if (ctl === 3) node.addClass("contested");
        else node.removeClass("contested");
      });

      // Edge color = blend of the two endpoint controller colors, width by total adjacent units.
      cy.edges().forEach((e) => {
        const s = e.source().data("tid") as number;
        const t = e.target().data("tid") as number;
        const cs = controllerColors[s];
        const ct = controllerColors[t];
        const color = cs === ct ? cs : mix(cs, ct);
        e.data("edgeColor", color);
        const heat = (totals[s] ?? 0) + (totals[t] ?? 0);
        e.data("edgeWidth", 1.2 + Math.min(heat * 0.06, 2.4));
      });

      // Highlight the territory the policy just acted on.
      const targetId = state.last_action?.target;
      cy.nodes(".last-target").removeClass("last-target");
      if (typeof targetId === "number") {
        const target = cy.getElementById(`t${targetId}`);
        if (target && !target.empty()) target.addClass("last-target");
      }
    });
  }, [state, map.name, map.territories]);

  return (
    <div className="map-canvas w-full h-full">
      <div ref={mountRef} className="w-full h-full" />
    </div>
  );
}
