# Import necessary modules from dash and plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objects as go
import pandas as pd

# Your remaining code follows here


# Example data preparation (replace with your actual data)
# Assuming df is your DataFrame with 'timestamp' (datetime), 'rating', and 'verified_purchase' columns

# Create sample data
# df = pd.DataFrame({
#     'timestamp': pd.date_range(start='1/1/2010', periods=1000, freq='M'),
#     'rating': np.random.randint(1, 6, size=1000),
#     'verified_purchase': np.random.choice([True, False], size=1000)
# })
# # Save to csv file
# df.to_csv('sample_data.csv', index=False)

# Load data from csv file
df = pd.read_csv('sample_data.csv')

# Extract year and month from timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month

# Dash application
app = web_app.Dash(__name__)

app.layout = html.Div([
    html.H1('Average Rating Over Years by Verification Status'),
    dcc.Checklist(
        id='month-checklist',
        options=[{'label': 'All', 'value': 'all'}] + [{'label': pd.to_datetime(month, format='%m').strftime('%B'), 'value': month} for month in range(1, 13)],
        value=['all'],
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Graph(id='rating-graph')
])

@app.callback(
    Output('rating-graph', 'figure'),
    Input('month-checklist', 'value')
)
def update_graph(selected_months):
    if 'all' in selected_months or not selected_months:
        filtered_df = df
    else:
        filtered_df = df[df['month'].isin(selected_months)]
    
    # Separate data for verified and non-verified users
    verified_data = filtered_df[filtered_df['verified_purchase'] == True]
    not_verified_data = filtered_df[filtered_df['verified_purchase'] == False]

    # Group by year and calculate average rating for each group
    avg_rating_verified = verified_data.groupby('year')['rating'].mean().reset_index()
    avg_rating_not_verified = not_verified_data.groupby('year')['rating'].mean().reset_index()
    avg_rating_total = filtered_df.groupby('year')['rating'].mean().reset_index()

    # Create traces for the yearly average ratings
    trace_verified_year = go.Scatter(x=avg_rating_verified['year'], y=avg_rating_verified['rating'],
                                     mode='lines+markers', name='Verified Users')
    trace_not_verified_year = go.Scatter(x=avg_rating_not_verified['year'], y=avg_rating_not_verified['rating'],
                                         mode='lines+markers', name='Not Verified Users')
    trace_total_year = go.Scatter(x=avg_rating_total['year'], y=avg_rating_total['rating'],
                                  mode='lines+markers', name='Total Average Rating', line=dict(dash='dash'))

    # Create layout for the plot
    layout = go.Layout(
        title='Average Rating Over Years by Verification Status',
        xaxis_title='Year',
        yaxis_title='Average Rating'
    )

    # Create figure and add traces to it
    fig = go.Figure(data=[trace_verified_year, trace_not_verified_year, trace_total_year], layout=layout)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
