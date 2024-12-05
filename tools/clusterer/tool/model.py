from dataclasses import dataclass


@dataclass
class RoadFeatures:
    sinuosity: float
    curvature_mean: float
    curvature_max: float
    curvature_std: float
    curvature_variability_index: float
    heading_change_mean: float
    heading_change_max: float
    heading_change_std: float
    straight_segments: int
    curved_segments: int
    total_segments: int
    mild_turns: int
    moderate_turns: int
    sharp_turns: int
    sharp_turn_ratio: float
    left_turns: int
    right_turns: int
    average_segment_length: float

    def to_vector(self) -> list:
        return [
            self.sinuosity,
            self.curvature_mean,
            self.curvature_max,
            self.curvature_std,
            self.curvature_variability_index,
            self.heading_change_mean,
            self.heading_change_max,
            self.heading_change_std,
            self.straight_segments,
            self.curved_segments,
            self.total_segments,
            self.mild_turns,
            self.moderate_turns,
            self.sharp_turns,
            self.sharp_turn_ratio,
            self.left_turns,
            self.right_turns,
            self.average_segment_length,
        ]

    def complexity(self) -> float:
        weights = [
            0.1,  # sinuosity
            0.15,  # curvature_mean
            0.15,  # curvature_max
            0.1,  # curvature_std
            0.1,  # curvature_variability_index
            0.05,  # heading_change_mean
            0.05,  # heading_change_max
            0.05,  # heading_change_std
            0.05,  # straight_segments
            0.05,  # curved_segments
            0.05,  # total_segments
            0.05,  # mild_turns
            0.05,  # moderate_turns
            0.1,  # sharp_turns
            0.1,  # sharp_turn_ratio
            0.05,  # left_turns
            0.05,  # right_turns
            0.1,  # average_segment_length
        ]

        # Calculate weighted sum as complexity score
        feature_vector = self.to_vector()
        complexity_score = sum(w * f for w, f in zip(weights, feature_vector))

        return complexity_score
