# -*- coding: utf-8 -*-
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Save outputs next to this script in an "outputs" subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
def out(filename):
    return os.path.join(OUTPUT_DIR, filename)

"""
Task 2: Constraint Satisfaction Problem - Map Colouring
=========================================================
(a) Australia - colour 5 mainland regions with {Blue, Red, Green}
    such that no two adjacent regions share the same colour.

(b) Nairobi - colour all 17 sub-counties using the fewest colours
    possible, with no two adjacent sub-counties sharing a colour.

Algorithm used: Backtracking search with
  • Forward checking (arc-consistency lite)
  • MRV  (Minimum Remaining Values)  heuristic
  • Degree heuristic (tie-break on MRV)
  • LCV  (Least Constraining Value)   heuristic
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════
# 1. Generic CSP Solver
# ══════════════════════════════════════════════════════════════════

class MapCSP:
    """Graph-colouring CSP solved with backtracking + heuristics."""

    def __init__(self, variables, neighbours, colours):
        self.variables  = variables            # list of region names
        self.neighbours = neighbours           # dict: region -> {adjacent regions}
        self.colours    = colours              # list of allowed colours
        self.assignment = {}
        self.nodes_explored = 0

    # ── helpers ──────────────────────────────────────────────────
    def consistent(self, region, colour, assignment):
        """Return True if assigning `colour` to `region` is consistent."""
        return all(
            assignment.get(nb) != colour
            for nb in self.neighbours[region]
        )

    def remaining_values(self, region, assignment, domains):
        """Count legal colours for `region` given current assignment."""
        return sum(
            1 for c in domains[region]
            if self.consistent(region, c, assignment)
        )

    def select_unassigned(self, assignment, domains):
        """MRV + degree heuristic."""
        unassigned = [v for v in self.variables if v not in assignment]
        # MRV: fewest legal values
        min_rv = min(self.remaining_values(v, assignment, domains) for v in unassigned)
        candidates = [v for v in unassigned
                      if self.remaining_values(v, assignment, domains) == min_rv]
        # degree tie-break: most constrained (most neighbours)
        return max(candidates, key=lambda v: len(self.neighbours[v]))

    def order_domain_values(self, region, assignment, domains):
        """LCV: least constraining value first."""
        def constraint_count(colour):
            return sum(
                1 for nb in self.neighbours[region]
                if nb not in assignment and colour in domains[nb]
            )
        return sorted(domains[region], key=constraint_count)

    def forward_check(self, region, colour, domains):
        """Remove `colour` from neighbours' domains; return pruned map."""
        pruned = {}
        for nb in self.neighbours[region]:
            if colour in domains[nb]:
                if nb not in pruned:
                    pruned[nb] = []
                pruned[nb].append(colour)
                domains[nb].remove(colour)
                if not domains[nb]:          # domain wipe-out
                    return None, pruned
        return domains, pruned

    def restore(self, domains, pruned):
        for region, cols in pruned.items():
            domains[region].extend(cols)

    # ── backtracking ─────────────────────────────────────────────
    def backtrack(self, assignment, domains):
        if len(assignment) == len(self.variables):
            return assignment

        region = self.select_unassigned(assignment, domains)
        for colour in self.order_domain_values(region, assignment, domains):
            self.nodes_explored += 1
            if self.consistent(region, colour, assignment):
                assignment[region] = colour
                new_domains, pruned = self.forward_check(region, colour,
                                                         {k: list(v) for k, v in domains.items()})
                if new_domains is not None:
                    result = self.backtrack(assignment, new_domains)
                    if result is not None:
                        return result
                del assignment[region]
        return None

    def solve(self):
        domains = {v: list(self.colours) for v in self.variables}
        self.nodes_explored = 0
        solution = self.backtrack({}, domains)
        return solution


# ══════════════════════════════════════════════════════════════════
# (a) Australia - 5 mainland regions, 3 colours
# ══════════════════════════════════════════════════════════════════

AUS_REGIONS = [
    "Western Australia (WA)",
    "Northern Territory (NT)",
    "South Australia (SA)",
    "Queensland (QLD)",
    "New South Wales (NSW)",
    # Note: Victoria and Tasmania are omitted so we stay at 5 regions
    # as specified. ACT is embedded in NSW.
]

AUS_NEIGHBOURS = {
    "Western Australia (WA)":     {"Northern Territory (NT)", "South Australia (SA)"},
    "Northern Territory (NT)":    {"Western Australia (WA)", "South Australia (SA)", "Queensland (QLD)"},
    "South Australia (SA)":       {"Western Australia (WA)", "Northern Territory (NT)",
                                   "Queensland (QLD)", "New South Wales (NSW)"},
    "Queensland (QLD)":           {"Northern Territory (NT)", "South Australia (SA)",
                                   "New South Wales (NSW)"},
    "New South Wales (NSW)":      {"South Australia (SA)", "Queensland (QLD)"},
}

AUS_COLOURS = ["Blue", "Red", "Green"]

# ─── Solve ────────────────────────────────────────────────────────
csp_aus = MapCSP(AUS_REGIONS, AUS_NEIGHBOURS, AUS_COLOURS)
sol_aus  = csp_aus.solve()

print("=" * 62)
print("(a) AUSTRALIA - Map Colouring (3 colours)")
print("=" * 62)
if sol_aus:
    for region, colour in sol_aus.items():
        print(f"  {region:<35} -> {colour}")
    print(f"\n  Nodes explored: {csp_aus.nodes_explored}")
    # verify
    valid = all(
        sol_aus[r] != sol_aus[nb]
        for r in AUS_REGIONS
        for nb in AUS_NEIGHBOURS[r]
        if nb in sol_aus
    )
    print(f"  Solution valid: {valid}")
else:
    print("  No solution found.")


# ── Visualise Australia ───────────────────────────────────────────
COLOUR_MAP = {
    "Blue": "#4A90D9", "Red": "#E05252", "Green": "#5CB85C",
    "Yellow": "#F0C040", "Purple": "#9B59B6", "Orange": "#E67E22",
}

# Approximate polygon coordinates (lon, lat) for each region
AUS_POLYS = {
    "Western Australia (WA)": [
        (114, -22), (114, -35), (129, -35), (129, -22), (114, -22)
    ],
    "Northern Territory (NT)": [
        (129, -16), (129, -26), (138, -26), (138, -16), (129, -16)
    ],
    "South Australia (SA)": [
        (129, -26), (129, -38), (141, -38), (141, -26), (138, -26), (138, -16), (129, -16), (129, -26)
    ],
    "Queensland (QLD)": [
        (138, -16), (138, -29), (154, -29), (154, -16), (138, -16)
    ],
    "New South Wales (NSW)": [
        (141, -29), (141, -37), (154, -37), (154, -29), (141, -29)
    ],
}

AUS_CENTRES = {
    "Western Australia (WA)": (121.5, -28),
    "Northern Territory (NT)": (133.5, -21),
    "South Australia (SA)": (135, -32),
    "Queensland (QLD)": (146, -22.5),
    "New South Wales (NSW)": (147.5, -33),
}

fig_a, ax_a = plt.subplots(figsize=(11, 7))
ax_a.set_facecolor("#D0E8F5")
fig_a.patch.set_facecolor("#1a1a2e")

for region, poly in AUS_POLYS.items():
    xs, ys = zip(*poly)
    colour = COLOUR_MAP[sol_aus[region]]
    ax_a.fill(xs, ys, color=colour, alpha=0.85, zorder=2)
    ax_a.plot(xs, ys, color="white", linewidth=1.5, zorder=3)
    cx, cy = AUS_CENTRES[region]
    # short label
    short = region.split("(")[1].rstrip(")")
    ax_a.text(cx, cy, short, ha="center", va="center",
              fontsize=11, fontweight="bold", color="white", zorder=4,
              path_effects=[pe.withStroke(linewidth=3, foreground="black")])

ax_a.set_xlim(112, 156)
ax_a.set_ylim(-40, -14)
ax_a.set_aspect("equal")
ax_a.set_title("(a) Australia - Map Colouring CSP\n(5 Regions · 3 Colours · No Adjacent Same Colour)",
               fontsize=13, fontweight="bold", color="white", pad=12)
ax_a.tick_params(colors="white")
for spine in ax_a.spines.values():
    spine.set_edgecolor("#555")

legend_patches = [mpatches.Patch(color=COLOUR_MAP[c], label=c) for c in AUS_COLOURS]
ax_a.legend(handles=legend_patches, loc="lower left", fontsize=10,
            facecolor="#222", labelcolor="white", edgecolor="#555")

plt.tight_layout()
plt.savefig(out("australia_colouring.png"), dpi=130, facecolor=fig_a.get_facecolor())
plt.close()
print(f"\nAustralia map saved -> {out('australia_colouring.png')}")


# ══════════════════════════════════════════════════════════════════
# (b) Nairobi - 17 Sub-Counties, Minimum Colours
# ══════════════════════════════════════════════════════════════════

# Nairobi's 17 administrative sub-counties and their adjacencies
# (based on the official county boundaries)
NAIROBI_SUBCOUNTIES = [
    "Westlands",      "Dagoretti North", "Dagoretti South", "Langata",
    "Kibra",          "Roysambu",        "Kasarani",        "Ruaraka",
    "Embakasi South", "Embakasi North",  "Embakasi Central","Embakasi East",
    "Embakasi West",  "Makadara",        "Kamukunji",       "Starehe",
    "Mathare",
]

# Adjacency graph based on geographic boundaries
NAIROBI_NEIGHBOURS = {
    "Westlands":        {"Roysambu", "Kasarani", "Starehe", "Dagoretti North"},
    "Dagoretti North":  {"Westlands", "Dagoretti South", "Kibra", "Starehe"},
    "Dagoretti South":  {"Dagoretti North", "Kibra", "Langata"},
    "Langata":          {"Dagoretti South", "Kibra", "Embakasi West", "Embakasi South"},
    "Kibra":            {"Dagoretti North", "Dagoretti South", "Langata",
                         "Embakasi West", "Starehe", "Kamukunji"},
    "Roysambu":         {"Westlands", "Kasarani", "Ruaraka"},
    "Kasarani":         {"Westlands", "Roysambu", "Ruaraka", "Embakasi North", "Mathare"},
    "Ruaraka":          {"Roysambu", "Kasarani", "Embakasi North", "Mathare", "Makadara"},
    "Embakasi South":   {"Langata", "Embakasi West", "Embakasi Central"},
    "Embakasi North":   {"Kasarani", "Ruaraka", "Embakasi Central", "Embakasi East", "Makadara"},
    "Embakasi Central": {"Embakasi South", "Embakasi North", "Embakasi East", "Embakasi West", "Makadara"},
    "Embakasi East":    {"Embakasi North", "Embakasi Central"},
    "Embakasi West":    {"Langata", "Kibra", "Embakasi South", "Embakasi Central", "Makadara", "Kamukunji"},
    "Makadara":         {"Ruaraka", "Embakasi North", "Embakasi Central",
                         "Embakasi West", "Kamukunji", "Starehe"},
    "Kamukunji":        {"Kibra", "Embakasi West", "Makadara", "Starehe"},
    "Starehe":          {"Westlands", "Dagoretti North", "Kibra",
                         "Kamukunji", "Makadara", "Mathare"},
    "Mathare":          {"Kasarani", "Ruaraka", "Starehe"},
}

# ─── Find minimum number of colours ───────────────────────────────
print("\n" + "=" * 62)
print("(b) NAIROBI - Map Colouring (minimum colours)")
print("=" * 62)

EXTRA_COLOURS = ["Blue", "Red", "Green", "Yellow", "Purple", "Orange"]

sol_nairobi = None
min_colours  = None
for n_col in range(1, len(EXTRA_COLOURS) + 1):
    colours_subset = EXTRA_COLOURS[:n_col]
    csp = MapCSP(NAIROBI_SUBCOUNTIES, NAIROBI_NEIGHBOURS, colours_subset)
    sol = csp.solve()
    if sol:
        sol_nairobi  = sol
        min_colours  = n_col
        nodes_exp    = csp.nodes_explored
        break

if sol_nairobi:
    colours_used = set(sol_nairobi.values())
    print(f"\n  Minimum colours required : {min_colours}")
    print(f"  Colours used             : {colours_used}")
    print(f"  Nodes explored           : {nodes_exp}")
    print("\n  Sub-county assignments:")
    for sc in NAIROBI_SUBCOUNTIES:
        print(f"    {sc:<22} -> {sol_nairobi[sc]}")
    # verify
    valid = all(
        sol_nairobi[r] != sol_nairobi[nb]
        for r in NAIROBI_SUBCOUNTIES
        for nb in NAIROBI_NEIGHBOURS[r]
        if nb in sol_nairobi
    )
    print(f"\n  Solution valid: {valid}")
else:
    print("  Could not find a solution.")


# ── Visualise Nairobi (schematic layout) ─────────────────────────
# Approximate (x, y) positions for a readable layout diagram
NAIROBI_POS = {
    "Westlands":        (2.5, 8.5),
    "Roysambu":         (5.0, 9.5),
    "Kasarani":         (7.5, 9.0),
    "Mathare":          (6.5, 7.5),
    "Ruaraka":          (8.0, 7.5),
    "Dagoretti North":  (1.5, 7.0),
    "Starehe":          (4.0, 7.0),
    "Kamukunji":        (5.5, 6.0),
    "Makadara":         (7.0, 6.0),
    "Dagoretti South":  (1.5, 5.5),
    "Kibra":            (3.0, 5.5),
    "Embakasi North":   (8.5, 5.5),
    "Embakasi West":    (4.5, 4.5),
    "Embakasi Central": (6.5, 4.5),
    "Embakasi East":    (8.5, 4.0),
    "Langata":          (2.5, 3.5),
    "Embakasi South":   (5.0, 3.0),
}

fig_n, ax_n = plt.subplots(figsize=(13, 10))
fig_n.patch.set_facecolor("#0d1117")
ax_n.set_facecolor("#0d1117")

# Draw edges first
drawn_edges = set()
for sc, nbs in NAIROBI_NEIGHBOURS.items():
    x1, y1 = NAIROBI_POS[sc]
    for nb in nbs:
        edge = tuple(sorted([sc, nb]))
        if edge not in drawn_edges:
            x2, y2 = NAIROBI_POS[nb]
            ax_n.plot([x1, x2], [y1, y2], color="#444", linewidth=1.5, zorder=1)
            drawn_edges.add(edge)

# Draw nodes
NODE_R = 0.7
for sc in NAIROBI_SUBCOUNTIES:
    x, y = NAIROBI_POS[sc]
    col  = COLOUR_MAP[sol_nairobi[sc]]
    circle = plt.Circle((x, y), NODE_R, color=col, zorder=2, alpha=0.9)
    ax_n.add_patch(circle)
    circle_border = plt.Circle((x, y), NODE_R, color="white", fill=False,
                                linewidth=1.5, zorder=3)
    ax_n.add_patch(circle_border)
    # Label - split long names
    label = sc.replace(" ", "\n") if len(sc) > 11 else sc
    ax_n.text(x, y, label, ha="center", va="center",
              fontsize=6.5, fontweight="bold", color="white", zorder=4,
              path_effects=[pe.withStroke(linewidth=2, foreground="black")])

ax_n.set_xlim(0, 10.5)
ax_n.set_ylim(1.5, 11)
ax_n.set_aspect("equal")
ax_n.axis("off")
ax_n.set_title(
    f"(b) Nairobi - 17 Sub-Counties Map Colouring\n"
    f"Minimum colours required: {min_colours}  |  Colours: {', '.join(sorted(set(sol_nairobi.values())))}",
    fontsize=13, fontweight="bold", color="white", pad=15
)

colours_used_list = sorted(set(sol_nairobi.values()))
legend_patches = [mpatches.Patch(color=COLOUR_MAP[c], label=c) for c in colours_used_list]
ax_n.legend(handles=legend_patches, loc="lower right", fontsize=10,
            facecolor="#222", labelcolor="white", edgecolor="#555", title="Colours",
            title_fontsize=9)

plt.tight_layout()
plt.savefig(out("nairobi_colouring.png"), dpi=130, facecolor=fig_n.get_facecolor())
plt.close()
print(f"\nNairobi map saved -> {out('nairobi_colouring.png')}")
print("\n✓ Both CSP tasks complete.")
