import json
import os
import streamlit as st
import altair as alt
import pandas as pd
from datetime import datetime
import numpy as np
from compute import MarkSixCompute

@st.cache_data(ttl=3600)
def load_data(data_source):
    data = pd.read_json(data_source)
    data['date'] = pd.to_datetime(data['date']).dt.date
    data['no'] = data['no'].apply(lambda x: [int(i) for i in x])
    data['sno'] = pd.to_numeric(data['sno'])
    return data

@st.cache_data(ttl=3600, show_spinner=False)
def load_color_definations(data_source):
    with open(data_source, encoding='utf-8') as file:
        data = json.load(file)
    return data

def remove_none_from_list(data):
    return list(filter(None, data))

ball_colors = load_color_definations(os.path.join(os.path.dirname(__file__), '../data/ball-colors.json'))

mark_six_compute = MarkSixCompute(ball_colors)

st.set_page_config(
    page_title="Mark Six Statistics",
    page_icon="🎱",
    layout="centered",
)
st.title('Mark Six Statistics')

st.markdown(
    'Mark Six (Chinese: 六合彩) is a lottery game organised by the Hong Kong Jockey Club. ' + \
    'The game is a 6-out-of-49 lottery-style game, with seven prize levels. ' + \
    'The winning numbers are selected automatically from a lottery machine that contains balls with numbers 1 to 49.'
)

# Load online so that we don't need to a re-deployment on data change
DATA_SOURCE = 'https://raw.githubusercontent.com/icelam/schedule-scrape-experiment/master/data/all.json'
mark_six_data = load_data(DATA_SOURCE)

dataset_last_updated = mark_six_data['date'].max().strftime('%Y/%m/%d');
number_of_records = len(mark_six_data.index);
st.caption(f'Dataset last updated on: {dataset_last_updated}, number of records: {number_of_records}')

tab1, tab2 = st.tabs(['Chart', 'Raw Data'])

with tab1:
    st.subheader('Occurrence of Balls')

    chart_option_column_1, chart_option_column_2, chart_option_column_3 = st.columns(3)

    min_data_date = mark_six_data['date'].min()
    max_data_date = mark_six_data['date'].max()
    date_range_to_display = chart_option_column_1.date_input(
        'Date Range',
        value=(min_data_date, max_data_date),
        min_value=min_data_date,
        max_value=max_data_date
    )

    group_by = chart_option_column_2.selectbox(
        'Group By',
        ('None', 'Odd / Even', 'Ball colors')
    )

    include_special_number = chart_option_column_3.selectbox(
        'Include special number',
        ('Yes', 'No')
    )

    # Insert spacing between option group and chart
    st.write('')

    filtered_mark_six_data = mark_six_data.copy()
    filtered_mark_six_data = filtered_mark_six_data[
        (filtered_mark_six_data['date'] >= date_range_to_display[0])
        & (filtered_mark_six_data['date'] <= date_range_to_display[-1])
    ]

    balls_summary = mark_six_compute.prepare_ball_summary(filtered_mark_six_data)
    id_interval_df = mark_six_compute.calculate_intervals(filtered_mark_six_data)

    if group_by == 'Ball colors':
        balls_summary = balls_summary.groupby(by='color').sum()
        balls_summary.drop(columns='ball', inplace=True)
        balls_summary.insert(0, 'color', balls_summary.index)
    elif group_by == 'Odd / Even':
        balls_summary['parity'] = balls_summary.ball.apply(lambda x: 'odd' if x % 2 == 0 else 'even')
        balls_summary = balls_summary.groupby(by='parity').sum()
        balls_summary.sort_values(by='parity', ascending=False, inplace=True)
        balls_summary.insert(0, 'odd_or_even', balls_summary.index)

        balls_summary.drop(columns='color', inplace=True)
        balls_summary.drop(columns='ball', inplace=True)

    balls_summary.reset_index(inplace=True, drop=True)

    # A customized version of st.bar_chart(histogram_values)
    # List of ptions: https://altair-viz.github.io/user_guide/customization.html
    chart_data = (
        alt.Chart(balls_summary)
            .transform_fold(remove_none_from_list([
                'count',
                'special_count' if include_special_number == 'Yes' else None
            ]))
            .mark_bar()
            .configure_axis(grid=False)
            .configure_view(strokeWidth=0)
            .properties(height=500)
    )

    if group_by == 'None':
        chart_data = (
            chart_data.encode(
                x=alt.X('ball:O', title='Balls'),
                y=alt.Y('value:Q', title='Occurrence'),
                color=alt.Color(
                    'color',
                    scale=alt.Scale(
                        domain=['red', 'blue', 'green'],
                        range=['lightcoral', 'royalblue', 'mediumseagreen']
                    ),
                    legend=None
                ),
                opacity=alt.Opacity(
                    'value:Q',
                    legend=None
                ),
                tooltip=remove_none_from_list([
                    alt.Tooltip('ball', title='Ball'),
                    alt.Tooltip('count', title='Occurrence'),
                    alt.Tooltip('special_count', title='Occurrence (Special)') if include_special_number == 'Yes' else None,
                    alt.Tooltip('total_count', title='Total Occurrence') if include_special_number == 'Yes' else None
                ])
            )
        )
    elif group_by == 'Ball colors':
        chart_data = (
            chart_data.encode(
                x=alt.X('color:N', title='Color'),
                y=alt.Y('value:Q', title='Occurrence'),
                color=alt.Color(
                    'color',
                    scale=alt.Scale(
                        domain=['red', 'blue', 'green'],
                        range=['lightcoral', 'royalblue', 'mediumseagreen']
                    ),
                    legend=None
                ),
                opacity=alt.Opacity(
                    'value:Q',
                    legend=None
                ),
                tooltip=remove_none_from_list([
                    alt.Tooltip('color', title='Color'),
                    alt.Tooltip('count', title='Occurrence'),
                    alt.Tooltip('special_count', title='Occurrence (Special)') if include_special_number == 'Yes' else None,
                    alt.Tooltip('total_count', title='Total Occurrence') if include_special_number == 'Yes' else None
                ])
            )
        )
    elif group_by == 'Odd / Even':
        chart_data = (
            chart_data.encode(
                x=alt.X('odd_or_even:N', title='Odd / Even', sort="descending"),
                y=alt.Y('value:Q', title='Occurrence'),
                color=alt.Color(
                    'odd_or_even',
                    scale=alt.Scale(
                        domain=['odd', 'even'],
                        range=['lightcoral', 'royalblue']
                    ),
                    legend=None
                ),
                opacity=alt.Opacity(
                    'value:Q',
                    legend=None
                ),
                tooltip=remove_none_from_list([
                    alt.Tooltip('odd_or_even', title='Odd / Even'),
                    alt.Tooltip('count', title='Occurrence'),
                    alt.Tooltip('special_count', title='Occurrence (Special)') if include_special_number == 'Yes' else None,
                    alt.Tooltip('total_count', title='Total Occurrence') if include_special_number == 'Yes' else None
                ])
            )
        )

    st.altair_chart(chart_data, use_container_width=True)

    if st.checkbox('Show data'):
        st.write(balls_summary)
    
    st.subheader('Interval of Balls Number Occurrence')
    filtered_mark_six_data['_id'] = range(1, len(filtered_mark_six_data) + 1)

    # Calculate intervals based on '_id'
    id_intervals = []

    for ball_no in range(1, 50):
        appearances = filtered_mark_six_data[filtered_mark_six_data['no'].apply(lambda x: ball_no in x)].index
        if len(appearances) > 1:
            latest_appearance = appearances[0]
            second_latest_appearance = appearances[1]
            id_interval = filtered_mark_six_data.loc[second_latest_appearance, '_id'] - filtered_mark_six_data.loc[latest_appearance, '_id']
            id_intervals.append({
                'ball_no': ball_no, 
                'id_interval': id_interval, 
                'latest_appearance_id': filtered_mark_six_data.loc[latest_appearance, '_id'], 
                'second_latest_appearance_id': filtered_mark_six_data.loc[second_latest_appearance, '_id'],
                'latest_appearance_date': filtered_mark_six_data.loc[latest_appearance, 'date'],
                'second_latest_appearance_date': filtered_mark_six_data.loc[second_latest_appearance, 'date']
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

    id_interval_df = pd.DataFrame(id_intervals)

    # Add color information to id_interval_df
    id_interval_df['color'] = id_interval_df['ball_no'].apply(lambda x: ball_colors[str(x)])

    # Display the new chart based on '_id' intervals with colors
    st.altair_chart(
        alt.Chart(id_interval_df).mark_bar().encode(
            x=alt.X('ball_no:O', title='Ball Number'),
            y=alt.Y('id_interval:Q', title='Interval (by _id)'),
            color=alt.Color(
                'color',
                scale=alt.Scale(
                    domain=['red', 'blue', 'green'],
                    range=['lightcoral', 'royalblue', 'mediumseagreen']
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip('ball_no:O', title='Ball Number'), 
                alt.Tooltip('id_interval:Q', title='Interval (by _id)'),
                alt.Tooltip('latest_appearance_id:Q', title='Latest Appearance ID'),
                alt.Tooltip('second_latest_appearance_id:Q', title='Second Latest Appearance ID'),
                alt.Tooltip('color', title='Ball Color')
            ]
        ).properties(
            title='Interval of Ball Number Occurrence (by _id)'
        ),
        use_container_width=True
    )

    if st.checkbox('Show ID interval data'):
        st.write(id_interval_df)

    st.subheader('Monte Carlo Simulation with Weight Convergence')
    
    num_weight_variations = st.slider('Number of Weight Variations', 5, 20, 10, step=1)
    simulations_per_weight = st.slider('Simulations per Weight Set', 10000, 100000, 50000, step=10000)
    
    selection = st.selectbox('Select Numbers to Generate', ('All Numbers Probability', 'Top Numbers Probability'))
    
    if st.button('Run Multiple Monte Carlo Simulations'):
        progress_bar = st.progress(0)
        all_results = []
        weight_combinations = []
        
        for i in range(num_weight_variations):
            # Generate weights following a normal distribution
            weights = np.random.random(size=3)  # Generate 3 random weights between 0 and 1
            weights = np.clip(weights, 0, 1)  # Ensure weights are between 0 and 1
            weights /= weights.sum()  # Normalize to sum to 1
            
            weight_combinations.append({
                'Occurrence Weight': weights[0],
                'Interval Weight': weights[1],
                'Color Weight': weights[2]
            })
            
            results = mark_six_compute.run_monte_carlo(
                balls_summary, 
                id_interval_df, 
                weights,
                simulations_per_weight
            )
            all_results.append(results)
            progress_bar.progress((i + 1) / (num_weight_variations))


        convergence_analysis, cv, common_top_10_count = mark_six_compute.analyze_simulation_results(
            all_results, 
            num_weight_variations
        )
        
        # Analyze convergence
        st.write('### Weight Combinations Used')
        weight_df = pd.DataFrame(weight_combinations)
        st.dataframe(
            weight_df.style.format({col: '{:.3f}' for col in weight_df.columns}),
            use_container_width=True
        )
        
        # Display convergence results
        st.write('### Convergence Analysis')
        
        # Visualization of overall frequencies
        conv_chart = alt.Chart(convergence_analysis).mark_bar().encode(
            x=alt.X('ball_no:O', title='Ball Number'),
            y=alt.Y('overall_frequency:Q', title='Overall Frequency'),
            color=alt.Color(
                'color:N',
                scale=alt.Scale(
                    domain=['red', 'blue', 'green'],
                    range=['lightcoral', 'royalblue', 'mediumseagreen']
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip('ball_no:O', title='Ball Number'),
                alt.Tooltip('overall_frequency:Q', title='Overall Frequency'),
                alt.Tooltip('occurrence_rate:Q', title='Occurrence Rate', format='.2%'),
                alt.Tooltip('color:N', title='Ball Color')
            ]
        ).properties(
            title=f'Overall Ball Frequency Across {num_weight_variations} Weight Variations',
            height=400
        )
        
        st.altair_chart(conv_chart, use_container_width=True)
        
        # Display top converged numbers
        st.write('### Top Converged Numbers')
        # Get top 5 numbers for each color
        colors = ['red', 'blue', 'green']
        top_by_color = []
        for color in colors:
            color_numbers = convergence_analysis[convergence_analysis['color'] == color]
            top_5 = color_numbers.nlargest(5, 'overall_frequency').copy()
            top_5['occurrence_rate'] = top_5['occurrence_rate'].apply(lambda x: f'{x:.2%}')
            top_by_color.append(top_5)
        
        # Combine results
        top_converged = pd.concat(top_by_color)
        
        st.dataframe(
            top_converged.style.background_gradient(subset=['overall_frequency'])
                            .format({'overall_frequency': '{:,.0f}'}),
            use_container_width=True
        )
        
        # Calculate stability metrics
        st.write('### Stability Analysis')
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Coefficient of Variation")
            st.metric("CV", f"{cv:.3f}")
            
        with col2:
            st.write("Top 10 Consistency")
            st.metric("Common Numbers", common_top_10_count)
        
        # Generate 6 numbers based on occurrence rates
        st.write('### Generated Numbers Based on Convergence Analysis')
        
        # Normalize occurrence rates to create probability distributions
        probabilities = convergence_analysis['occurrence_rate'].values
        ball_numbers = convergence_analysis['ball_no'].values
        
        # Create probability distribution from top converged numbers
        top_probabilities = top_converged['occurrence_rate'].apply(lambda x: float(x.strip('%'))/100).values
        top_probabilities = top_probabilities / top_probabilities.sum()  # Normalize
        top_ball_numbers = top_converged['ball_no'].values
        
        # Create comparison dataframe of probabilities
        all_numbers_prob_comparison = pd.DataFrame({
            'Ball Number': ball_numbers,
            'Probability Type': ['All Numbers']*len(ball_numbers),
            'Probability': probabilities
        })

        top_numbers_prob_comparison = pd.DataFrame({
            'Ball Number': top_ball_numbers,
            'Probability Type': ['Top Numbers']*len(top_ball_numbers),
            'Probability': top_probabilities
        })
        st.write("### Probability Distributions")
        st.dataframe(
            all_numbers_prob_comparison.style.format({'Probability': '{:.3%}'})
                          .background_gradient(subset=['Probability']),
            use_container_width=True
        )

        st.dataframe(
            top_numbers_prob_comparison.style.format({'Probability': '{:.3%}'})
                          .background_gradient(subset=['Probability']),
            use_container_width=True
        )
        

        if selection == 'Top Numbers':
            # Generate 6 unique numbers using the probability distribution
            st.write("### Suggested Numbers Based on Top Numbers")
            # Ensure ball_numbers and top_probabilities have same length by using only top numbers
            generated_numbers = np.random.choice(
                top_ball_numbers,  # Use top_ball_numbers instead of ball_numbers
                size=6,
                replace=False,
                p=top_probabilities
            )
            generated_numbers = np.sort(generated_numbers)
        else:
            st.write("### Suggested Numbers Based on All Numbers")
            # Generate 6 unique numbers using the probability distribution
            generated_numbers = np.random.choice(
                ball_numbers,
                size=6,
                replace=False,
                p=probabilities
            )
            generated_numbers = np.sort(generated_numbers)
        # Display generated numbers
        st.write("Suggested numbers based on convergence analysis:")
        number_cols = st.columns(6)
        for i, num in enumerate(generated_numbers):
            ball_color = convergence_analysis.loc[convergence_analysis['ball_no'] == num, 'color'].iloc[0]
            number_cols[i].metric(
                "Ball " + str(i+1),
                int(num),
                help=f"Color: {ball_color}"
            )
        


with tab2:
    st.write(mark_six_data)


