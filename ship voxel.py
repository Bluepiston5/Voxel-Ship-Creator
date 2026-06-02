"""
Ship Voxel Builder
Builds a hollow voxel ship hull from parameters, exports to JSON for browser preview
and .schem for Minecraft.
"""

import numpy as np
import json
import struct
import gzip
from pathlib import Path

# ─── Wool Block IDs (Minecraft 1.12 legacy for .schem compatibility) ───
WOOL_COLORS = {
    "white":        (35, 0),
    "orange":       (35, 1),
    "magenta":      (35, 2),
    "light_blue":   (35, 3),
    "yellow":       (35, 4),
    "lime":         (35, 5),
    "pink":         (35, 6),
    "gray":         (35, 7),
    "light_gray":   (35, 8),
    "cyan":         (35, 9),
    "purple":       (35, 10),
    "blue":         (35, 11),
    "brown":        (35, 12),
    "green":        (35, 13),
    "red":          (35, 14),
    "black":        (35, 15),
}

# ─── Default Ship Parameters ───────────────────────────────────────────
DEFAULT_PARAMS = {
    "länge":    200,
    "breite":   30,
    "tiefgang": 8,
    "freibord": 10,

    "bug": {
        "kiel":         -2,
        "wulst":        +2,
        "wasserlinie":  -1,
        "seite":         0,
    },
    "mitte": {
        "kiel":          0,
        "wulst":        +3,
        "wasserlinie":  -2,
        "seite":         0,
    },
    "heck": {
        "kiel":         -1,
        "wulst":        +1,
        "wasserlinie":  -1,
        "seite":         0,
    },

    "farben": {
        "rumpf":        "gray",
        "kiel":         "black",
        "deck":         "brown",
        "wasserlinie":  "white",
        "aufbau_1":     "light_gray",
        "aufbau_2":     "yellow",
        "aufbau_3":     "orange",
        "aufbau_4":     "lime",
        "aufbau_5":     "cyan",
        "aufbau_6":     "pink",
        "aufbau_7":     "purple",
        "aufbau_8":     "red",
    }
}


class ShipBuilder:
    def __init__(self, params=None):
        self.p = params or DEFAULT_PARAMS
        self.L = self.p["länge"]
        self.B = self.p["breite"]
        self.D = self.p["tiefgang"]
        self.F = self.p["freibord"]
        self.H = self.D + self.F  # total height

        # voxel grid: stores color name or None
        # axes: x=length, y=height, z=width (half, then mirrored)
        self.half_z = self.B // 2 + 4  # extra for dents
        self.grid = np.full((self.L, self.H + 2, self.B + 8), None, dtype=object)

    # ─── Segment blend factor ──────────────────────────────────────────
    def _segment_params(self, x_frac):
        """Interpolate dent parameters at position x_frac (0.0 to 1.0)"""
        bug = self.p["bug"]
        mid = self.p["mitte"]
        heck = self.p["heck"]

        def lerp(a, b, t):
            return a + (b - a) * t

        def smooth(t):
            return t * t * (3 - 2 * t)  # smoothstep

        if x_frac < 0.25:
            t = smooth(x_frac / 0.25)
            return {k: lerp(bug[k], mid[k], t) for k in bug}
        elif x_frac < 0.75:
            t = smooth((x_frac - 0.25) / 0.50)
            return {k: lerp(mid[k], mid[k], t) for k in mid}
        else:
            t = smooth((x_frac - 0.75) / 0.25)
            return {k: lerp(mid[k], heck[k], t) for k in mid}

    # ─── Hull cross-section at position x ──────────────────────────────
    def _hull_width_at(self, x_frac):
        """Returns half-width of hull at this position (tapers at bow/stern)"""
        # Elliptical taper
        if x_frac < 0.15:
            t = x_frac / 0.15
            return self.B / 2 * np.sqrt(max(0, t))
        elif x_frac > 0.85:
            t = (1.0 - x_frac) / 0.15
            return self.B / 2 * np.sqrt(max(0, t))
        else:
            return self.B / 2

    # ─── Build the hull ────────────────────────────────────────────────
    def build_hull(self):
        """Fill voxel grid with hull (half ship, Z=0 is centerline)"""
        farben = self.p["farben"]
        total_z = self.B + 8  # grid width

        for x in range(self.L):
            x_frac = x / (self.L - 1)
            seg = self._segment_params(x_frac)
            half_w = self._hull_width_at(x_frac)

            # Height at this x: tapers at bow
            local_h = self.H
            if x_frac < 0.05:
                local_h = max(2, int(self.H * (x_frac / 0.05)))
            elif x_frac > 0.95:
                local_h = max(2, int(self.H * ((1 - x_frac) / 0.05)))

            for z in range(int(half_w) + 4):
                z_frac = z / (half_w + 0.001)

                # Apply wulst (bulge) dent - affects mid-height outward
                wulst = seg["wulst"]
                wasserlinie = seg["wasserlinie"]
                kiel_dent = seg["kiel"]

                # Effective half-width with dents
                if z_frac < 1.0:
                    # Bulge: push out at waterline area
                    bulge_factor = np.exp(-((z_frac - 0.7) ** 2) / 0.1)
                    eff_w = half_w + wulst * bulge_factor

                    # Tumblehome: push in at top
                    tumble_factor = np.exp(-((z_frac - 1.0) ** 2) / 0.05)
                    eff_w += wasserlinie * tumble_factor
                else:
                    eff_w = half_w + wasserlinie * np.exp(-((z_frac - 1.0)**2)/0.05)

                # Stack blocks vertically
                for y in range(local_h):
                    y_frac = y / local_h

                    # Keel dent: V-shape at bottom
                    keel_reduction = abs(kiel_dent) * (1 - z_frac) * (1 - y_frac)
                    if kiel_dent < 0:
                        if z < abs(kiel_dent) * (1 - y_frac):
                            continue  # skip = V keel shape

                    # Only place block if within effective width
                    if z <= eff_w:
                        # Determine color
                        if y == 0:
                            color = farben["kiel"]
                        elif y == local_h - 1:
                            color = farben["deck"]
                        elif y == self.D:  # waterline
                            color = farben["wasserlinie"]
                        else:
                            color = farben["rumpf"]

                        self.grid[x, y, z] = color

        return self

    # ─── Hollow out (keep only shell) ──────────────────────────────────
    def hollow(self):
        """Remove interior blocks, keep only outer shell"""
        new_grid = np.full_like(self.grid, None)
        L, H, Z = self.grid.shape

        for x in range(L):
            for y in range(H):
                for z in range(Z):
                    if self.grid[x, y, z] is None:
                        continue
                    # Check if any neighbor is air
                    is_surface = False
                    for dx, dy, dz in [
                        (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)
                    ]:
                        nx, ny, nz = x+dx, y+dy, z+dz
                        if (nx < 0 or nx >= L or ny < 0 or ny >= H or
                            nz < 0 or nz >= Z):
                            is_surface = True
                            break
                        if self.grid[nx, ny, nz] is None:
                            is_surface = True
                            break
                    if is_surface:
                        new_grid[x, y, z] = self.grid[x, y, z]

        self.grid = new_grid
        return self

    # ─── Mirror Z axis ─────────────────────────────────────────────────
    def mirror(self):
        """Mirror half-ship to full ship (Z axis)"""
        L, H, Z = self.grid.shape
        full_z = Z * 2 - 1
        full_grid = np.full((L, H, full_z), None, dtype=object)

        for x in range(L):
            for y in range(H):
                for z in range(Z):
                    c = self.grid[x, y, z]
                    if c is not None:
                        full_grid[x, y, z] = c
                        full_grid[x, y, full_z - 1 - z] = c

        self.grid = full_grid
        return self

    # ─── Export to JSON for browser viewer ─────────────────────────────
    def to_json(self, path="ship.json"):
        L, H, Z = self.grid.shape
        blocks = []

        for x in range(L):
            for y in range(H):
                for z in range(Z):
                    c = self.grid[x, y, z]
                    if c is not None:
                        blocks.append([x, y, z, c])

        data = {
            "dimensions": {"x": L, "y": H, "z": Z},
            "blocks": blocks,
            "params": self.p,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Exported {len(blocks)} blocks to {path}")
        return path

    # ─── Export to .schem (WorldEdit schematic) ────────────────────────
    def to_schem(self, path="ship.schem"):
        """Export as legacy .schematic (MCEdit/WorldEdit format)"""
        L, H, Z = self.grid.shape

        # Build block arrays
        blocks = bytearray(L * H * Z)
        data_arr = bytearray(L * H * Z)

        for x in range(L):
            for y in range(H):
                for z in range(Z):
                    c = self.grid[x, y, z]
                    idx = y * L * Z + z * L + x  # YZX order
                    if c is not None and c in WOOL_COLORS:
                        block_id, meta = WOOL_COLORS[c]
                        blocks[idx] = block_id
                        data_arr[idx] = meta

        # Write NBT
        nbt = self._make_schematic_nbt(L, H, Z, bytes(blocks), bytes(data_arr))
        with gzip.open(path, "wb") as f:
            f.write(nbt)
        print(f"Exported schematic to {path}")
        return path

    def _make_schematic_nbt(self, width, height, length, blocks, data):
        """Minimal NBT writer for .schematic format"""
        def tag_string(name, value):
            n = name.encode("utf-8")
            v = value.encode("utf-8")
            return b'\x08' + struct.pack(">H", len(n)) + n + struct.pack(">H", len(v)) + v

        def tag_short(name, value):
            n = name.encode("utf-8")
            return b'\x02' + struct.pack(">H", len(n)) + n + struct.pack(">h", value)

        def tag_byte_array(name, data):
            n = name.encode("utf-8")
            return b'\x07' + struct.pack(">H", len(n)) + n + struct.pack(">i", len(data)) + data

        def tag_list_empty(name):
            n = name.encode("utf-8")
            return b'\x09' + struct.pack(">H", len(n)) + n + b'\x0a' + struct.pack(">i", 0)

        payload = (
            tag_string("Materials", "Alpha") +
            tag_short("Width", width) +
            tag_short("Height", height) +
            tag_short("Length", length) +
            tag_byte_array("Blocks", blocks) +
            tag_byte_array("Data", data) +
            tag_list_empty("Entities") +
            tag_list_empty("TileEntities") +
            b'\x00'  # TAG_End
        )

        name = b"Schematic"
        header = b'\x0a' + struct.pack(">H", len(name)) + name
        return header + payload


# ─── CLI Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    params_file = sys.argv[1] if len(sys.argv) > 1 else None

    if params_file and Path(params_file).exists():
        with open(params_file) as f:
            params = json.load(f)
    else:
        params = DEFAULT_PARAMS

    print("Building ship hull...")
    builder = ShipBuilder(params)
    builder.build_hull()
    print("Hollowing...")
    builder.hollow()
    print("Mirroring...")
    builder.mirror()

    out_dir = Path(params_file).parent if params_file else Path(".")
    builder.to_json(str(out_dir / "ship.json"))
    builder.to_schem(str(out_dir / "ship.schem"))
    print("Done!")
