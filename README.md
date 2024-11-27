# Mark Six Data Visualization

A data visualization tool for Hong Kong Mark Six lottery data analysis.

## Features

### Basic Statistics
- Historical data visualization
- Ball occurrence frequency analysis
- Grouping by colors and odd/even numbers
- Raw data display

### Advanced Analysis

#### Interval Analysis
- Tracks intervals between ball appearances
- Visualizes occurrence patterns over time
- Color-coded ball tracking
- Detailed interval metrics for each ball

#### Monte Carlo Simulation with Weight Convergence

The application implements an advanced Monte Carlo simulation system to analyze ball patterns and generate predictions:

**Simulation Parameters:**
- Number of Weight Variations (5-20): Controls diversity of probability distributions
- Simulations per Weight Set (10,000-100,000): Determines statistical significance
- Selection Mode: Choose between All Numbers or Top Numbers probability distributions

**How It Works:**

1. **Weight Generation**
   - Creates multiple sets of random weights for:
     - Occurrence frequency
     - Interval patterns
     - Color distribution

2. **Convergence Analysis**
   - Runs multiple simulations with different weight combinations
   - Analyzes convergence patterns across simulations
   - Calculates stability metrics:
     - Coefficient of Variation (CV)
     - Top 10 number consistency

3. **Probability Distribution**
   - Generates two types of probability distributions:
     - All Numbers: Uses complete historical data
     - Top Numbers: Focuses on most frequent numbers
   - Visualizes distributions with color-coding
   - Provides detailed probability metrics

4. **Number Generation**
   - Offers two generation modes:
     - All Numbers Probability: Uses full distribution
     - Top Numbers Probability: Uses concentrated distribution
   - Generates sets of 6 unique numbers
   - Displays results with color indicators

**Visualization Features:**
- Color-coded bar charts for frequency analysis
- Probability distribution comparisons
- Weight combination analysis
- Generated number displays with color indicators

## Installation & Usage

[Original installation instructions...]

## Data Source

The data is sourced from Hong Kong Jockey Club's historical Mark Six results.

## License

[Original license information...]
