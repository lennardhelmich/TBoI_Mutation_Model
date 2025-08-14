import math
from collections import deque
import numpy as np
from PIL import Image
from tboi_bitmap import TBoI_Bitmap, EntityType
from constants_2 import Constants

class Inverted_Fitness_Function:
    """
    'Schlecht aber spielbar' ohne Poop/Fire-Anteil.
    Harte Constraints: Traversability + gültige Spawns.
    Ziele: Anti-Symmetrie, hohe Variation, Topologie-Badness, Umweg-Faktor, Anti-Balance, Gegnernähe.
    functionValue:
    [ total, inv_sym, inv_var, topo_bad, detour, anti_balance, enemy_prox ]
    """

    # Gewichte (kannst du per ctor-Argument überschreiben)
    W_INV_SYMMETRY = Constants.FITNESS_WEIGHT_SYMMETRY
    W_INV_VARIATION = Constants.FITNESS_WEIGHT_VARIATION
    W_TOPOLOGY     = Constants.INVERTED_FITNESS_WEIGHT_TOPOLOGY
    W_DETOUR       = Constants.INVERTED_FITNESS_WEIGHT_DETOUR
    W_ENEMY_PROX   = Constants.INVERTED_FITNESS_WEIGHT_EPROX
    W_ENEMY_DIFF   = Constants.FITNESS_WEIGHT_ENEMIES
    W_BITMAP_CHANGES = Constants.FITNESS_WEIGHT_CHANGES

    # Optional: Mindestabstand Gegner ↔ Spawn (Manhattan) als Sicherheitspuffer
    MIN_SAFE_ENEMY_SPAWN_DISTANCE = 2

    def __init__(self, startBitmap, resultBitmap, weights: dict | None = None):
        start_tboi = TBoI_Bitmap()
        start_tboi.bitmap = Image.fromarray(np.array(startBitmap, dtype=np.uint8), mode="L")
        start_tboi.create_graph_out_of_bitmap()

        result_tboi = TBoI_Bitmap()
        result_tboi.bitmap = Image.fromarray(np.array(resultBitmap, dtype=np.uint8), mode="L")
        result_tboi.create_graph_out_of_bitmap()

        self.startBitmap = start_tboi
        self.resultBitmap = result_tboi
        self.functionValue = []
        self._reachable_cache: set[tuple[int,int]] = set()

        if weights:
            for k, v in weights.items():
                setattr(self, k, float(v))

    # ---------- Hilfen ----------
    def _neighbors4(self, x, y):
        return ((x-1,y), (x+1,y), (x,y-1), (x,y+1))

    def _manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    # ---------- Harte Constraints ----------
    def check_every_traversability(self):
        doors, enemies = [], []
        for x in range(self.resultBitmap.width):
            for y in range(self.resultBitmap.height):
                pv = self.resultBitmap.bitmap.getpixel((x, y))
                ent = self.resultBitmap.get_entity_id_with_pixel_value(pv)
                if ent == EntityType.DOOR:
                    doors.append((x, y))
                elif ent == EntityType.ENTITY:
                    enemies.append((x, y))
        if not doors:
            return False

        first = doors[0]
        targets = set(doors + enemies)
        self._reachable_cache.clear()
        return self.resultBitmap.is_path_existent(
            first, targets, visited_out=self._reachable_cache, exhaustive=True
        )

    def _poop_fire_over_limit(self, limit: int = 10) -> bool:
        """True, wenn Anzahl von Poop+Fire > limit."""
        bm = self.resultBitmap
        count = 0
        # Schneller: Pixel einmal laden
        px = bm.bitmap.load()
        for y in range(bm.height):
            for x in range(bm.width):
                ent = bm.get_entity_id_with_pixel_value(px[x, y])
                if ent == EntityType.POOP or ent == EntityType.FIRE:
                    count += 1
                    if count > limit:   # early exit
                        return True
        return False

    def check_spawn(self):
        bm = self.resultBitmap.bitmap
        for (x, y) in [(7,1), (7,7), (1,4), (13,4)]:
            pv = bm.getpixel((x,y))
            ent = self.resultBitmap.get_entity_id_with_pixel_value(pv)
            if ent != EntityType.FREE_SPACE:
                return False
        return True

    # ---------- Ziele („schlecht“ aber spielbar) ----------
    def inverted_pixel_variation(self):
        bmap = self.resultBitmap
        w, h = bmap.bitmap.size
        matches = total = 0
        for y in range(1, h-1):
            for x in range(1, w-1):
                cur = bmap.bitmap.getpixel((x, y))
                for nx, ny in bmap.get_all_neighbors((x, y)):
                    if cur == bmap.bitmap.getpixel((nx, ny)):
                        matches += 1
                    total += 1
        if total == 0:
            return 0.0
        return 1.0 - (matches/total)

    def inverted_symmetry(self):
        bm = self.resultBitmap.bitmap
        w, h = bm.size

        def inv_vert():
            comp = match = 0
            for i in range(1, h-1):
                for j in range(1, math.ceil(w/2)):
                    comp += 1
                    if bm.getpixel((j,i)) == bm.getpixel((w-1-j,i)):
                        match += 1
            return 1.0 - (match/comp if comp else 0.0)

        def inv_horz():
            comp = match = 0
            for i in range(1, math.ceil(h/2)):
                for j in range(1, w-1):
                    comp += 1
                    if bm.getpixel((j,i)) == bm.getpixel((j,h-1-i)):
                        match += 1
            return 1.0 - (match/comp if comp else 0.0)

        def inv_cen():
            comp = match = 0
            for i in range(1, h-1):
                for j in range(1, w-1):
                    comp += 1
                    if bm.getpixel((j,i)) == bm.getpixel((w-1-j,h-1-i)):
                        match += 1
            return 1.0 - (match/comp if comp else 0.0)

        return (inv_vert() + inv_horz() + inv_cen()) / 3.0

    def _build_reachable_degree_map(self):
        deg = {}
        for (x, y) in self._reachable_cache:
            d = 0
            for nx, ny in self._neighbors4(x, y):
                if (nx, ny) in self._reachable_cache:
                    d += 1
            deg[(x, y)] = d
        return deg

    def topology_badness(self):
        if not self._reachable_cache:
            return 0.0
        deg = self._build_reachable_degree_map()
        n = len(deg)
        if n == 0:
            return 0.0
        dead_ends = sum(1 for v in deg.values() if v == 1)
        corridors = sum(1 for v in deg.values() if v == 2)
        junctions = sum(1 for v in deg.values() if v >= 3)
        score = 0.6*(dead_ends/n) + 0.5*(corridors/n) - 0.4*(junctions/n)
        return max(0.0, min(1.0, score))

    def _bfs_geodesic_len(self, start, goal):
        if start == goal:
            return 0
        from collections import deque
        q = deque([start])
        dist = {start: 0}
        while q:
            x, y = q.popleft()
            d = dist[(x, y)]
            for nx, ny in self._neighbors4(x, y):
                if (nx, ny) in self._reachable_cache and (nx, ny) not in dist:
                    dist[(nx, ny)] = d + 1
                    if (nx, ny) == goal:
                        return d + 1
                    q.append((nx, ny))
        return None

    def detour_score(self):
        doors = []
        for x in range(self.resultBitmap.width):
            for y in range(self.resultBitmap.height):
                pv = self.resultBitmap.bitmap.getpixel((x, y))
                if self.resultBitmap.get_entity_id_with_pixel_value(pv) == EntityType.DOOR:
                    doors.append((x, y))
        if len(doors) < 2 or not self._reachable_cache:
            return 0.0

        base = doors[0]
        ratios = []
        for d in doors[1:]:
            man = max(1, self._manhattan(base, d))
            geo = self._bfs_geodesic_len(base, d)
            if geo is None:
                continue
            ratios.append(geo / man)

        if not ratios:
            return 0.0

        avg = sum(ratios) / len(ratios)
        return max(0.0, min(1.0, (avg - 1.0) / 3.0))  # 1→0, >=4→1

    def enemy_proximity_score(self):
        bm = self.resultBitmap
        enemy_val = bm.get_pixel_value_with_entity_id(EntityType.ENTITY)
        spawns = [(7,1), (7,7), (1,4), (13,4)]
        enemies = []
        for x in range(bm.width):
            for y in range(bm.height):
                if bm.bitmap.getpixel((x, y)) == enemy_val:
                    enemies.append((x, y))
        if not enemies:
            return 0.0

        dists = [min(self._manhattan(e, s) for s in spawns) for e in enemies]
        avg = sum(dists) / len(dists)

        if avg < self.MIN_SAFE_ENEMY_SPAWN_DISTANCE:
            return 0.0  # Sicherheits-Puffer

        w, h = bm.width, bm.height
        maxd = max(w, h)
        return 1.0 - min(1.0, avg / maxd)
    
    def enemy_difference_value(self):
        """
        1.0, wenn Gegneranzahl sich höchstens um MAX_NUMBER_FREE_ENEMY_CHANGES ändert,
        sonst linear abfallend gemäß VALUE_REDUCTION_PER_ENEMY.
        """
        enemy_value = self.resultBitmap.get_pixel_value_with_entity_id(EntityType.ENTITY)
        enemies_start = list(self.startBitmap.bitmap.getdata()).count(enemy_value)
        enemies_now   = list(self.resultBitmap.bitmap.getdata()).count(enemy_value)
        difference = abs(enemies_now - enemies_start)

        if difference <= Constants.MAX_NUMBER_FREE_ENEMY_CHANGES:
            return 1.0
        else:
            over = difference - Constants.MAX_NUMBER_FREE_ENEMY_CHANGES
            return max(0.0, 1.0 - Constants.VALUE_REDUCTION_PER_ENEMY * over)
        
    def bitmap_changes(self):
        list_prev = list(self.startBitmap.bitmap.getdata())
        list_now = list(self.resultBitmap.bitmap.getdata())
        total_count = (self.resultBitmap.bitmap.width-2) * (self.resultBitmap.bitmap.height-2)
        difference_in_pixels = sum(1 for p1, p2 in zip(list_prev, list_now) if p1 != p2)
        difference_percent = (difference_in_pixels/total_count)*100
        if(difference_percent > (Constants.TARGETED_BITMAP_DIFFERENCE*2)):
            return 0
        else:
            return (1-(abs(difference_percent-Constants.TARGETED_BITMAP_DIFFERENCE)*(1/Constants.TARGETED_BITMAP_DIFFERENCE)))

    # ---------- Aggregation ----------
    def calc_fitness_function(self):
        if not self.check_every_traversability() or not self.check_spawn():
            self.functionValue = [0, 0, 0, 0, 0, 0, 0, 0]
            return

        if self._poop_fire_over_limit(10):
            self.functionValue = [0, 0, 0, 0, 0, 0, 0, 0]
            return

        inv_sym = self.inverted_symmetry()
        inv_var = self.inverted_pixel_variation()
        topo   = self.topology_badness()
        detour = self.detour_score()
        eprox  = self.enemy_proximity_score()
        bitmap_changes = self.bitmap_changes()
        ediff  = self.enemy_difference_value()

        wsum = (self.W_INV_SYMMETRY + self.W_INV_VARIATION + self.W_TOPOLOGY +
                self.W_DETOUR + self.W_BITMAP_CHANGES + self.W_ENEMY_PROX + self.W_ENEMY_DIFF)

        value = (self.W_INV_SYMMETRY*inv_sym +
                 self.W_INV_VARIATION*inv_var +
                 self.W_TOPOLOGY*topo +
                 self.W_DETOUR*detour +
                 self.W_ENEMY_PROX*eprox + 
                 self.W_BITMAP_CHANGES*bitmap_changes +
                 self.W_ENEMY_DIFF*ediff)

        total = value / wsum if wsum > 0 else 0.0
        self.functionValue = [total, inv_sym, inv_var, topo, detour, eprox, ediff, bitmap_changes]


if __name__ == "__main__":
    path = "Bitmaps/InitRooms/bitmap_32.bmp"
    bitmap = Image.open(path)
    fitness = Inverted_Fitness_Function(bitmap, bitmap)
    fitness.calc_fitness_function()
    print(fitness.functionValue)
