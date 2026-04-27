import { useEffect, useRef } from "react";
import cytoscape, { Core, ElementDefinition } from "cytoscape";
import type { LiveState, MapInfo } from "../lib/api";

type Props = {
  map: MapInfo;
  state: LiveState | null;
};

const NATION_COLOR: Record<number, string> = {
  0: "#c95757", // invader
  1: "#5b8eda", // defender
  2: "#6c6f7a", // neutral
  3: "#3a3d44", // contested
};

export default function MapGraph({ map, state }: Props) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!mountRef.current) return;
    const elements: ElementDefinition[] = [
      ...map.territories.map((t) => ({
        data: {
          id: `t${t.id}`,
          label: t.name,
          home_of: t.home_of,
          strategic: t.strategic_value,
        },
      })),
      ...map.edges.map(([u, v]) => ({
        data: { id: `e${u}-${v}`, source: `t${u}`, target: `t${v}` },
      })),
    ];

    const cy = cytoscape({
      container: mountRef.current,
      elements,
      layout: { name: "cose", animate: false, idealEdgeLength: () => 90, padding: 24 },
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "background-color": "#1d1f25",
            "border-width": 1,
            "border-color": "#2a2d35",
            "text-valign": "center",
            "text-halign": "center",
            "font-family": "Inter, system-ui, sans-serif",
            "font-size": 10,
            "font-weight": 500,
            color: "#e6e7ea",
            width: 64,
            height: 64,
            "text-wrap": "wrap",
            "text-max-width": "60",
          },
        },
        {
          selector: "edge",
          style: {
            "line-color": "#2a2d35",
            width: 1,
            "curve-style": "straight",
          },
        },
      ],
      wheelSensitivity: 0.2,
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
    cy.batch(() => {
      map.territories.forEach((t) => {
        const ctl = state.controller[t.id];
        const inv = state.invader_units[t.id] ?? 0;
        const def = state.defender_units[t.id] ?? 0;
        const neu = state.neutral_units[t.id] ?? 0;
        const node = cy.getElementById(`t${t.id}`);
        if (!node) return;
        node.style("border-color", NATION_COLOR[ctl] ?? "#2a2d35");
        node.style("border-width", 2);
        node.style(
          "background-color",
          ctl === 3 ? "#191b21" : `${NATION_COLOR[ctl]}22`
        );
        const totalUnits = inv + def + neu;
        node.style("width", 56 + Math.min(totalUnits * 2.2, 28));
        node.style("height", 56 + Math.min(totalUnits * 2.2, 28));
        node.data(
          "label",
          `${t.name}\nI ${inv.toFixed(0)}  D ${def.toFixed(0)}  N ${neu.toFixed(0)}`
        );
        if (ctl === 3) node.addClass("contested");
        else node.removeClass("contested");
      });
    });
  }, [state, map.name, map.territories]);

  return <div ref={mountRef} className="w-full h-full" />;
}
