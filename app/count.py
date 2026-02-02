class PeopleCounter:
    def __init__(self, line_y):
        self.line_y = line_y

        self.total_entered = 0
        self.total_exited = 0

        self.males = 0
        self.females = 0

        self.prev_center_y = {}      # id -> prev_y
        self.counted_direction = {}  # id -> "entered"/"exited"

    def update(self, tracked_objects, gender_lookup):
        """
        tracked_objects: list[(id, box)]
        gender_lookup(tid) -> "male"/"female"/None
        """
        current_count = len(tracked_objects)

        for tid, box in tracked_objects:
            x1, y1, x2, y2 = box
            cy = (y1 + y2) / 2

            prev = self.prev_center_y.get(tid)
            self.prev_center_y[tid] = cy

            if prev is None:
                continue

            crossed_down = prev < self.line_y and cy >= self.line_y
            crossed_up = prev > self.line_y and cy <= self.line_y

            if tid not in self.counted_direction:
                if crossed_down:
                    self.total_entered += 1
                    self.counted_direction[tid] = "entered"
                elif crossed_up:
                    self.total_exited += 1
                    self.counted_direction[tid] = "exited"

                # count gender once
                g = gender_lookup(tid)
                if g == "male":
                    self.males += 1
                elif g == "female":
                    self.females += 1

        return {
            "current_count": current_count,
            "total_entered": self.total_entered,
            "total_exited": self.total_exited,
            "males": self.males,
            "females": self.females,
        }
