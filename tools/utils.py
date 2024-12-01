import numpy as np
from shapely.geometry import LineString, Point
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import logging

@dataclass
class RoadAnalysis:
    """Store enhanced road analysis results"""
    curvature_profile: np.ndarray
    total_length: float
    mean_curvature: float
    max_curvature: float
    turn_count: int
    complexity_score: float
    segment_lengths: np.ndarray
    angle_changes: np.ndarray
    max_angle_change: float

class RoadAnalyzer:
    def __init__(self):
        self.curvature_threshold = 0.08 
        self.min_turn_angle = np.pi / 6 
        self.logger = logging.getLogger(__name__)
        
    def compute_curvature_profile(self, road_points: List[Tuple[float, float]], sample_distance: float = 1.0) -> np.ndarray:
        """Enhanced curvature computation with denser sampling"""
        road_line = LineString(road_points)
        road_length = road_line.length
        
        
        curvature_profile = np.zeros(max(int(road_length / sample_distance), 2))
        
        for i in range(len(curvature_profile)):
            s = i * sample_distance
            if s < sample_distance or s > road_length - sample_distance:
                continue
                
            # Get three points with smaller intervals
            pt_before = road_line.interpolate(s - sample_distance)
            pt_current = road_line.interpolate(s)
            pt_after = road_line.interpolate(s + sample_distance)
            
            # Calculate vectors
            vec1 = np.array([pt_current.x - pt_before.x, pt_current.y - pt_before.y])
            vec2 = np.array([pt_after.x - pt_current.x, pt_after.y - pt_current.y])
            
            vec1_norm = np.linalg.norm(vec1)
            vec2_norm = np.linalg.norm(vec2)
            
            if vec1_norm > 1e-6 and vec2_norm > 1e-6: 
                vec1_normalized = vec1 / vec1_norm
                vec2_normalized = vec2 / vec2_norm
                
                # Calculate angle 
                dot_product = np.clip(np.dot(vec1_normalized, vec2_normalized), -1.0, 1.0)
                angle = np.arccos(dot_product)
                
                # curvature calculation
                curvature_profile[i] = angle / sample_distance
                
        return curvature_profile
    
    def compute_segment_properties(self, road_points: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute segment lengths and angle changes between segments"""
        points = np.array(road_points)
        segments = points[1:] - points[:-1]
        
        # Compute segment lengths
        segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
        
        # Compute angle changes between segments
        segment_vectors = segments / segment_lengths[:, np.newaxis]
        dot_products = np.sum(segment_vectors[1:] * segment_vectors[:-1], axis=1)
        angle_changes = np.arccos(np.clip(dot_products, -1.0, 1.0))
        
        return segment_lengths, angle_changes
    
    def analyze_road(self, road_points: List[Tuple[float, float]]) -> RoadAnalysis:
        """Enhanced road analysis with multiple features"""
        if len(road_points) < 2:
            self.logger.warning("Road has too few points for analysis")
            return RoadAnalysis(
                curvature_profile=np.array([]),
                total_length=0,
                mean_curvature=0,
                max_curvature=0,
                turn_count=0,
                complexity_score=0,
                segment_lengths=np.array([]),
                angle_changes=np.array([]),
                max_angle_change=0
            )
            
        road_line = LineString(road_points)
        curvature_profile = self.compute_curvature_profile(road_points)
        segment_lengths, angle_changes = self.compute_segment_properties(road_points)
        
        # metrics calculation
        total_length = road_line.length
        mean_curvature = np.mean(curvature_profile[curvature_profile > 0])
        max_curvature = np.max(curvature_profile)
        turn_count = np.sum(angle_changes > self.min_turn_angle)
        max_angle_change = np.max(angle_changes) if len(angle_changes) > 0 else 0
        
        # complexity score calculation
        length_factor = np.clip(total_length / 100, 0.5, 2.0)  # Normalized by typical length
        curvature_factor = np.clip(mean_curvature / 0.1, 0, 3.0)  # Normalized by typical curvature
        turn_density = turn_count / max(total_length / 50, 1)  # Turns per 50 distance units
        
        complexity_score = (
            0.4 * curvature_factor +          # Base curvature importance
            0.3 * turn_density +              # Turn density importance
            0.2 * (max_curvature / 0.3) +     # Maximum curvature importance
            0.1 * length_factor               # Length factor
        )
        
        return RoadAnalysis(
            curvature_profile=curvature_profile,
            total_length=total_length,
            mean_curvature=mean_curvature,
            max_curvature=max_curvature,
            turn_count=turn_count,
            complexity_score=complexity_score,
            segment_lengths=segment_lengths,
            angle_changes=angle_changes,
            max_angle_change=max_angle_change
        )
    
    def calculate_road_similarity(self, road1_analysis: RoadAnalysis, road2_analysis: RoadAnalysis) -> float:
        """Enhanced similarity calculation using multiple features"""
        # Normalize length difference
        length_diff = abs(road1_analysis.total_length - road2_analysis.total_length) / max(
            max(road1_analysis.total_length, road2_analysis.total_length), 1.0
        )
        
        # Normalize curvature profiles to same length for comparison
        len1 = len(road1_analysis.curvature_profile)
        len2 = len(road2_analysis.curvature_profile)
        target_len = min(len1, len2)
        
        if target_len > 0:
            profile1 = road1_analysis.curvature_profile[:target_len]
            profile2 = road2_analysis.curvature_profile[:target_len]
            
            # Calculate curvature profile similarity
            profile_diff = np.mean(np.abs(profile1 - profile2))
            
            # Calculate feature-based similarity
            feature_similarity = 1 - np.mean([
                abs(road1_analysis.mean_curvature - road2_analysis.mean_curvature) / max(road1_analysis.mean_curvature, 0.001),
                abs(road1_analysis.turn_count - road2_analysis.turn_count) / max(road1_analysis.turn_count, 1),
                length_diff,
                abs(road1_analysis.max_angle_change - road2_analysis.max_angle_change) / (np.pi + 0.001)
            ])
            
            # Weighted combination of similarities
            similarity = (
                0.4 * (1 - profile_diff) +     # Curvature profile similarity
                0.4 * feature_similarity +      # Feature-based similarity
                0.2 * (1 - length_diff)        # Length similarity
            )
            
            return np.clip(similarity, 0, 1)
        
        return 0.0

    def visualize_road(self, road_points: List[Tuple[float, float]], analysis: RoadAnalysis = None, save_path: str = None):
        """Enhanced visualization with more metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot road layout
        road_x, road_y = zip(*road_points)
        ax1.plot(road_x, road_y, 'b-', label='Road')
        ax1.set_aspect('equal')
        ax1.set_title('Road Layout')
        ax1.grid(True)
        ax1.legend()
        
        if analysis:
            # Plot curvature profile
            ax2.plot(analysis.curvature_profile, 'r-', label='Curvature')
            ax2.axhline(y=self.curvature_threshold, color='g', linestyle='--', label='Turn Threshold')
            ax2.set_title(f'Curvature Profile\nComplexity Score: {analysis.complexity_score:.2f}')
            ax2.grid(True)
            ax2.legend()
            
            # Plot segment lengths
            ax3.plot(analysis.segment_lengths, 'b-', label='Segment Lengths')
            ax3.set_title(f'Segment Lengths\nTotal Length: {analysis.total_length:.1f}')
            ax3.grid(True)
            ax3.legend()
            
            # Plot angle changes
            ax4.plot(np.degrees(analysis.angle_changes), 'm-', label='Angle Changes')
            ax4.axhline(y=np.degrees(self.min_turn_angle), color='g', linestyle='--', label='Turn Threshold')
            ax4.set_title(f'Angle Changes\nTurn Count: {analysis.turn_count}')
            ax4.grid(True)
            ax4.legend()
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()