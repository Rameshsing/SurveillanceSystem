class LineCounter:
    def __init__(self, line_position):
        self.line_y = line_position
        self.count_in = 0
        self.count_out = 0
        self.previous_positions = {}

    def update(self, tracked_objects):
        for obj_id, (cx, cy) in tracked_objects.items():
            if obj_id in self.previous_positions:
                prev_y = self.previous_positions[obj_id]
                if prev_y < self.line_y and cy >= self.line_y:
                    self.count_in += 1
                elif prev_y > self.line_y and cy <= self.line_y:
                    self.count_out += 1
            self.previous_positions[obj_id] = cy
