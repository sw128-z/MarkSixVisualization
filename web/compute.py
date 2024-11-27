import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class MarkSixCompute:
    def __init__(self, ball_colors: Dict[str, str]):
        self.ball_colors = ball_colors

    def prepare_ball_summary(self, mark_six_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare ball summary statistics from raw data"""
        balls_summary = pd.DataFrame(list(range(1, 50)), columns=['ball'])
        
        balls_count = mark_six_data['no'].explode().value_counts().sort_index().to_frame()
        balls_count.insert(0, 'ball', balls_count.index)
        
        special_ball_count = mark_six_data['sno'].value_counts().sort_index().to_frame()
        special_ball_count.insert(0, 'ball', special_ball_count.index)
        
        balls_summary = balls_summary.merge(balls_count, on='ball', how='left')
        balls_summary = balls_summary.merge(special_ball_count, on='ball', how='left')
        
        balls_summary = balls_summary.rename(columns={'count_x': 'count', 'count_y': 'special_count'})
        balls_summary['special_count'].fillna(0, inplace=True)
        balls_summary['count'].fillna(0, inplace=True)
        
        balls_summary.insert(3, 'total_count', balls_summary['count'] + balls_summary['special_count'])
        balls_summary.insert(4, 'color', balls_summary['ball'].apply(lambda x: self.ball_colors[str(x)]))
        
        return balls_summary

    def calculate_intervals(self, mark_six_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate intervals between ball appearances"""
        mark_six_data = mark_six_data.copy()
        mark_six_data['_id'] = range(1, len(mark_six_data) + 1)
        
        id_intervals = []
        for ball_no in range(1, 50):
            appearances = mark_six_data[mark_six_data['no'].apply(lambda x: ball_no in x)].index
            if len(appearances) > 1:
                latest_appearance = appearances[0]
                second_latest_appearance = appearances[1]
                id_interval = mark_six_data.loc[second_latest_appearance, '_id'] - mark_six_data.loc[latest_appearance, '_id']
                id_intervals.append({
                    'ball_no': ball_no, 
                    'id_interval': id_interval, 
                    'latest_appearance_id': mark_six_data.loc[latest_appearance, '_id'], 
                    'second_latest_appearance_id': mark_six_data.loc[second_latest_appearance, '_id'],
                    'latest_appearance_date': mark_six_data.loc[latest_appearance, 'date'],
                    'second_latest_appearance_date': mark_six_data.loc[second_latest_appearance, 'date']
                })
            else:
                id_intervals.append({
                    'ball_no': ball_no, 
                    'id_interval': None, 
                    'latest_appearance_id': None, 
                    'second_latest_appearance_id': None,
                    'latest_appearance_date': None,
                    'second_latest_appearance_date': None
                })
        
        interval_df = pd.DataFrame(id_intervals)
        interval_df['color'] = interval_df['ball_no'].apply(lambda x: self.ball_colors[str(x)])
        return interval_df

    def calculate_combined_probabilities(
        self, 
        balls_summary_df: pd.DataFrame, 
        interval_df: pd.DataFrame, 
        weights: Tuple[float, float, float]
    ) -> np.ndarray:
        """Calculate combined probabilities based on multiple factors"""
        occurrence_weight, interval_weight, color_weight = weights
        
        # Historical occurrence probability
        occurrence_probs = balls_summary_df['total_count'] / balls_summary_df['total_count'].sum()
        
        # Interval-based probability
        max_interval = interval_df['id_interval'].max()
        intervals = interval_df['id_interval'].fillna(max_interval)
        interval_probs = (max_interval - intervals) / (max_interval - intervals).sum()
        # Color probability
        color_counts = balls_summary_df.groupby('color')['total_count'].sum()
        color_probs_dict = (color_counts / color_counts.sum()).to_dict()
        
        color_probs = []
        for i in range(1, 50):
            ball_color = self.ball_colors[str(i)]
            color_probs.append(color_probs_dict[ball_color])
        color_probs = np.array(color_probs)
        
        # Combine probabilities
        combined_probs = (
            occurrence_weight * occurrence_probs.values +
            interval_weight * interval_probs.values +
            color_weight * color_probs
        )
        
        return combined_probs / combined_probs.sum()

    def run_monte_carlo(
        self, 
        balls_summary_df: pd.DataFrame, 
        interval_df: pd.DataFrame, 
        weights: Tuple[float, float, float], 
        num_simulations: int,
        batch_size: int = 1000
    ) -> np.ndarray:
        """Run Monte Carlo simulation with given parameters"""
        probabilities = self.calculate_combined_probabilities(
            balls_summary_df, 
            interval_df, 
            weights
        )
        
        simulated_results = []
        for i in range(0, num_simulations, batch_size):
            batch_end = min(i + batch_size, num_simulations)
            batch_count = batch_end - i
            
            batch_results = [
                np.random.choice(
                    range(1, 50), 
                    size=6, 
                    replace=False, 
                    p=probabilities
                ) for _ in range(batch_count)
            ]
            simulated_results.extend([sorted(draw) for draw in batch_results])
            
        return np.array(simulated_results)

    def analyze_simulation_results(
        self, 
        all_results: List[np.ndarray], 
        num_weight_variations: int
    ) -> Tuple[pd.DataFrame, float, int]:
        """Analyze results from multiple Monte Carlo simulations"""
        all_numbers = np.concatenate([res.flatten() for res in all_results])
        overall_frequency = pd.Series(all_numbers).value_counts().sort_index()
        
        convergence_analysis = pd.DataFrame({
            'ball_no': range(1, 50),
            'overall_frequency': overall_frequency.reindex(range(1, 50)).fillna(0),
            'color': [self.ball_colors[str(i)] for i in range(1, 50)]
        })
        
        convergence_analysis['occurrence_rate'] = (
            convergence_analysis['overall_frequency'] / 
            convergence_analysis['overall_frequency'].sum()
        )
        
        # Calculate stability metrics
        cv = convergence_analysis['overall_frequency'].std() / convergence_analysis['overall_frequency'].mean()
        
        # Calculate common top 10 numbers
        top_10_sets = [
            set(pd.Series(res.flatten()).value_counts().nlargest(10).index)
            for res in all_results
        ]
        common_top_10 = set.intersection(*top_10_sets)
        
        return convergence_analysis, cv, len(common_top_10) 

    def run_monte_carlo_with_zero_weights(
        self, 
        balls_summary_df: pd.DataFrame, 
        interval_df: pd.DataFrame, 
        initial_weights: Tuple[float, float, float], 
        num_simulations: int = 4
    ) -> List[np.ndarray]:
        """Run additional Monte Carlo simulations with one weight set to zero."""
        all_results = []
        
        for weight_index in range(3):
            modified_weights = list(initial_weights)
            modified_weights[weight_index] = 0  # Set one weight to zero
            modified_weights = np.array(modified_weights) / np.sum(modified_weights)  # Normalize
            
            for _ in range(num_simulations):
                results = self.run_monte_carlo(balls_summary_df, interval_df, modified_weights, 10000)
                all_results.append(results)
        
        return all_results 
