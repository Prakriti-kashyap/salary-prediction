import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import plotly.express as px

# Set Streamlit page config
st.set_page_config(layout='wide', page_title='Startup Funding Analysis')


# Load and clean data
# Example of loading your data correctly
# Load and clean data
# Example of loading your data correctly
@st.cache_data
def load_data():
    df = pd.read_csv('startup_cleaned.csv')  # Replace with your file path
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df is not None and not df.empty:
        df.columns = df.columns.astype(str).str.strip().str.lower()
    else:
        st.error("DataFrame is not loaded correctly. Please check the file path or data source.")

    # Ensure 'date' column is in datetime form
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows where 'date' is NaT
    df = df.dropna(subset=['date'])

    # Extract Year-Month in 'YYYY-MM' format
    df['YearMonth'] = df['date'].dt.strftime('%Y-%m')

    # Extract Year and Month separately
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    return df  # ✅ Correctly indented inside the function


df = load_data()

# Predefined latitude and longitude mapping (Only required if missing in dataset)
city_coordinates = {
    "Bangalore": {"latitude": 12.9716, "longitude": 77.5946},
    "Mumbai": {"latitude": 19.0760, "longitude": 72.8777},
    "Delhi": {"latitude": 28.7041, "longitude": 77.1025},
    "Hyderabad": {"latitude": 17.3850, "longitude": 78.4867},
    "Chennai": {"latitude": 13.0827, "longitude": 80.2707},
}


# Overall Analysis
def load_overall_analysis(overall_analysis):
    st.title('📊 Overall Startup Funding Analysis')

    # Assign the passed DataFrame to df
    df = overall_analysis.copy()

    # --- 🧹 Data Cleaning & Preprocessing ---
    st.info("🔎 Checking and Cleaning Data...")

    # Clean column names
    df.columns = df.columns.astype(str).str.strip().str.lower()

    # Required columns for analysis
    required_columns = {'amount', 'year', 'month', 'city', 'sector', 'startup', 'round', 'investors', 'date'}
    missing_columns = required_columns - set(df.columns)
    required_columns = {'amount', 'year', 'month', 'city', 'startup', 'round', 'investors', 'date'}  # Removed 'sector'
    # Assuming 'vertical' is the sector column
    # If your sector column has a different name, replace 'vertical' with that name

    if 'vertical' not in df.columns:
        st.error("🚨 Error: The DataFrame is missing the 'vertical' column (or your sector column name).")
        st.stop()
    else:
        required_columns.add('vertical')  # Add 'vertical' to required columns

    missing_columns = required_columns - set(df.columns)

    # Check if any required columns are missing
    if missing_columns:
        st.error(f"🚨 Error: Missing required columns: {', '.join(missing_columns)}")
        st.stop()

    # Convert 'amount' to numeric and handle missing values
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)

    # --- Add further analysis and visualization here ---
    st.success("✅ Data cleaned successfully! Ready for analysis.")

    # Removed the problematic line as 'date' is already processed in load_data
    # df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.dropna(subset=['date', 'amount'])
    df['yearmonth'] = df['date'].dt.to_period('M').astype(str)

    # --- 📈 Investment Statistics ---
    st.subheader('📈 Investment Statistics')

    # Key stats
    total_investment = df['amount'].sum()
    max_investment = df['amount'].max()
    avg_investment = df['amount'].mean()
    num_startups = df['startup'].nunique()

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label='💸 Total Investment (Cr)', value=f"{total_investment:,.2f}")
    col2.metric(label='🚀 Maximum Investment (Cr)', value=f"{max_investment:,.2f}")
    col3.metric(label='📊 Average Investment (Cr)', value=f"{avg_investment:,.2f}")
    col4.metric(label='🏢 Funded Startups', value=num_startups)

    # --- 📆 Month-over-Month & Year-over-Year Analysis ---
    st.subheader("📆 Month-over-Month & Year-over-Year Investment Patterns")

    min_year, max_year = df['year'].min(), df['year'].max()
    selected_years = st.slider("🔍 Select Year Range", min_value=min_year, max_value=max_year,
                               value=(min_year, max_year))

    # Filter data by selected years
    filtered_df = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]
    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected year range.")
        return

    # Group by YearMonth
    investment_trend = filtered_df.groupby('yearmonth')['amount'].sum().reset_index()
    fig_mom_yoy = px.line(investment_trend, x='yearmonth', y='amount', title="📊 Investment Trend Over Time")
    st.plotly_chart(fig_mom_yoy, use_container_width=True)

    # --- 🏙️ Top Cities for Startup Funding ---
    st.subheader('🏙️ Top Cities for Startup Funding')
    top_cities = df.groupby('city')['amount'].sum().nlargest(10).reset_index()
    fig_cities = px.bar(
        top_cities, x='city', y='amount', title='🏙️ Top 10 Cities for Startup Funding',
        color='amount', color_continuous_scale='Plasma'
    )
    st.plotly_chart(fig_cities, use_container_width=True)

    # --- 🚀 Most Funded Startups ---
    st.subheader('🚀 Most Funded Startups')
    most_funded = df.groupby('startup')['amount'].sum().nlargest(10).reset_index()
    fig_startups = px.bar(
        most_funded, x='startup', y='amount', title='🏆 Top 10 Most Funded Startups',
        color='amount', color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_startups, use_container_width=True)

    # --- 📈 Year-over-Year Growth ---
    st.subheader('📈 Year-over-Year Growth')
    yoy_growth = df.groupby('year')['amount'].sum().reset_index()
    fig_yoy_overall = px.line(
        yoy_growth, x='year', y='amount', markers=True, title='📊 Overall Year-over-Year Growth'
    )
    st.plotly_chart(fig_yoy_overall, use_container_width=True)


# Investor Analysis (Unchanged)
# Corrected function to accept investor as argument
def load_investor_analysis(df, investor):
    st.title(f'🔍 Investor Analysis: {investor}')

    # Filter DataFrame based on the selected investor
    investor_df = df[df['investors'].str.contains(investor, na=False, case=False)].copy()

    if investor_df.empty:
        st.warning('❗ No data available for this investor.')
        return

    # 🎛️ --- Add Interactive Filters ---
    with st.sidebar:
        st.subheader('🔍 Filters')

        # Date Range Filter
        start_date, end_date = st.date_input(
            'Select Date Range',
            [investor_df['date'].min(), investor_df['date'].max()]
        )
        investor_df = investor_df[(investor_df['date'] >= pd.to_datetime(start_date)) &
                                  (investor_df['date'] <= pd.to_datetime(end_date))]

        # Sector/Vertical Filter
        selected_sector = st.multiselect('Filter by Sector', investor_df['vertical'].unique())
        if selected_sector:
            investor_df = investor_df[investor_df['vertical'].isin(selected_sector)]

        # Round Filter
        selected_round = st.multiselect('Filter by Investment Round', investor_df['round'].unique())
        if selected_round:
            investor_df = investor_df[investor_df['round'].isin(selected_round)]

        # City Filter
        selected_city = st.multiselect('Filter by City', investor_df['city'].unique())
        if selected_city:
            investor_df = investor_df[investor_df['city'].isin(selected_city)]

    # 🎯 --- Summary Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric(label='💸 Total Investment (Cr)', value=f"{investor_df['amount'].sum():,.2f}")
    col2.metric(label='🚀 Number of Startups Funded', value=investor_df['startup'].nunique())
    col3.metric(label='📈 Number of Rounds', value=investor_df['round'].nunique())

    # 📊 --- Top Startups Funded ---
    st.subheader('🏆 Top Startups Funded by the Investor')
    top_startups = investor_df.groupby('startup')['amount'].sum().nlargest(10).reset_index()
    fig_startups = px.bar(
        top_startups, x='startup', y='amount', title='Top 10 Startups Funded by the Investor',
        color='amount', color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_startups, use_container_width=True)

    # 📅 --- Funding Over Time ---
    st.subheader('📊 Funding Over Time by the Investor')
    funding_over_time = investor_df.groupby('YearMonth')['amount'].sum().reset_index()
    fig_funding = px.line(
        funding_over_time, x='YearMonth', y='amount', markers=True,
        title='Funding Over Time by the Investor'
    )
    st.plotly_chart(fig_funding, use_container_width=True)

    # 📚 --- Investment Rounds Breakdown ---
    st.subheader('📈 Investment Rounds Breakdown')
    rounds_data = investor_df.groupby('round')['amount'].sum().reset_index()
    fig_rounds = px.pie(
        rounds_data, names='round', values='amount', title='Investment Rounds Breakdown'
    )
    st.plotly_chart(fig_rounds, use_container_width=True)

    # 🌇 --- Top Cities Where Investor is Active ---
    st.subheader('🌆 Top Cities Where Investor is Active')
    top_cities = investor_df.groupby('city')['amount'].sum().nlargest(5).reset_index()
    fig_cities = px.bar(
        top_cities, x='city', y='amount', title='Top Cities with Most Investment',
        color='amount', color_continuous_scale='Turbo'
    )
    st.plotly_chart(fig_cities, use_container_width=True)

    # 📊 --- Sector Distribution ---
    st.subheader('📡 Sector Distribution of Investments')
    sector_distribution = investor_df.groupby('vertical')['amount'].sum().reset_index()
    fig_sector = px.pie(
        sector_distribution, values='amount', names='vertical', title='Sector Distribution'
    )
    st.plotly_chart(fig_sector, use_container_width=True)

    # 📊 --- Year-over-Year Growth ---
    st.subheader('📊 Year-over-Year (YoY) Growth in Investment')
    yoy_investments = investor_df.groupby('year')['amount'].sum().pct_change().dropna() * 100
    yoy_investments = yoy_investments.reset_index(name='growth')
    fig_yoy = px.bar(
        yoy_investments, x='year', y='growth', title='YoY Growth in Investment (%)',
        color='growth', color_continuous_scale='Magma'
    )
    st.plotly_chart(fig_yoy, use_container_width=True)

    # 🔥 --- Investment Heatmap (Year vs. Month) ---
    st.subheader('🌡️ Investment Heatmap (Year vs. Month)')
    heatmap_data = investor_df.groupby(['year', 'month'])['amount'].sum().unstack().fillna(0)
    fig_heatmap, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=0.5, ax=ax)
    ax.set_title('Investment Heatmap by Year and Month')
    st.pyplot(fig_heatmap)

    # 📈 --- Investment Trends by Sector ---
    st.subheader('📊 Investment Trends by Sector')
    sector_trends = investor_df.groupby(['vertical', 'YearMonth'])['amount'].sum().unstack().fillna(0)
    fig_sector_trends, ax = plt.subplots(figsize=(12, 6))
    sector_trends.T.plot(ax=ax)
    plt.title('Investment Trends by Sector')
    plt.xlabel('Year-Month')
    plt.ylabel('Amount (Cr)')
    plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig_sector_trends)

    # 🕒 --- Most Recent Investments ---
    st.subheader('📅 Most Recent Investments')
    st.dataframe(
        investor_df[['date', 'startup', 'vertical', 'city', 'round', 'amount']].sort_values(by='date',
                                                                                            ascending=False).head(10)
    )


# Startup Analysis with Dynamic Filtering
def load_startup_analysis(df, startup_name):
    st.title(f'Startup Analysis: {startup_name}')
    startup_df = df[df['startup'] == startup_name]

    # Ensure 'month' column exists to create 'yearmonth'
    if 'month' not in startup_df.columns:
        st.error("Month data is missing. Please ensure there is a 'month' column in the dataset.")
        return

        # Create 'yearmonth' column if it doesn't exist
    startup_df['yearmonth'] = pd.to_datetime(startup_df[['year', 'month']].assign(day=1))

    # Dynamic Year Range Selection
    if startup_df['year'].dropna().empty:
        st.warning("No valid year data available for this startup.")
        return

    min_year, max_year = int(startup_df['year'].min()), int(startup_df['year'].max())

    if min_year == max_year:
        min_year = max_year - 1  # Set a safe range

    selected_years = st.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    # Additional Filters
    selected_rounds = st.multiselect("Select Investment Rounds", options=startup_df['round'].unique(),
                                     default=startup_df['round'].unique())
    selected_cities = st.multiselect("Select Cities", options=startup_df['city'].dropna().unique(),
                                     default=startup_df['city'].dropna().unique())

    # Filter Data Based on Selected Filters
    filtered_startup_df = startup_df[
        (startup_df['year'] >= selected_years[0]) &
        (startup_df['year'] <= selected_years[1]) &
        (startup_df['round'].isin(selected_rounds)) &
        (startup_df['city'].isin(selected_cities))
        ]

    if filtered_startup_df.empty:
        st.warning("No data available for the selected filters.")
        return

    st.dataframe(filtered_startup_df)

    # Funding Over Time
    st.subheader("Funding Over Time")
    funding_over_time = filtered_startup_df.groupby('yearmonth')['amount'].sum().reset_index()
    fig_funding = px.line(
        funding_over_time, x='yearmonth', y='amount', markers=True, title=f"Funding Over Time for {startup_name}"
    )
    st.plotly_chart(fig_funding, use_container_width=True)

    #
    # Investment Rounds Breakdown
    st.subheader('Investment Rounds Breakdown')
    rounds_data = filtered_startup_df.groupby('round')['amount'].sum().sort_values(ascending=False).reset_index()
    fig_rounds = px.bar(
        rounds_data, x='round', y='amount', title='Investment Rounds Breakdown', color='amount',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_rounds, use_container_width=True)

    # Top Investors
    st.subheader('Top Investors for the Startup')
    top_investors = filtered_startup_df['investors'].str.split(',').explode().str.strip().value_counts().head(
        5).reset_index()
    top_investors.columns = ['investor', 'count']
    fig_investors = px.bar(
        top_investors, x='investor', y='count', title='Top Investors', labels={'investor': 'Investor', 'count': 'Count'}
    )
    st.plotly_chart(fig_investors, use_container_width=True)

    # Competitor Comparison
    st.subheader('Comparison with Other Startups in the Same Sector')
    sector = filtered_startup_df['vertical'].iloc[0] if not filtered_startup_df.empty else None
    if sector:
        sector_df = df[df['vertical'] == sector]
        sector_funding = sector_df.groupby('startup')['amount'].sum().nlargest(10).reset_index()
        fig_sector = px.bar(
            sector_funding, x='startup', y='amount', title='Top 10 Startups in the Sector',
            color='amount', color_continuous_scale='Plasma'
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    # Year-over-Year Growth
    st.subheader('Year-over-Year Growth of Startup')
    yoy_growth = filtered_startup_df.groupby('year')['amount'].sum().reset_index()
    fig_yoy = px.line(
        yoy_growth, x='year', y='amount', markers=True, title=f'Year-over-Year Growth of {startup_name}'
    )
    st.plotly_chart(fig_yoy, use_container_width=True)


# Feedback Section (Unchanged)
# Sidebar Menu
st.sidebar.title('🚀 Startup Funding Analysis')
option = st.sidebar.selectbox(
    '🔍 Select Analysis Type',
    ['Overall Analysis', 'Startup', 'Investor']
)

# Handle Startup Analysis
if option == 'Startup':
    selected_startup = st.sidebar.selectbox(
        '🏢 Select Startup', sorted(df['startup'].dropna().unique())
    )
    if st.sidebar.button('📊 Show Startup Details'):
        load_startup_analysis(df, selected_startup)

# Handle Overall Analysis
elif option == 'Overall Analysis':
    load_overall_analysis(df)

# Handle Investor Analysis
elif option == 'Investor':
    # Initialize session state for investor selection if not already set
    if 'selected_investor' not in st.session_state:
        st.session_state.selected_investor = None

    # Generate sorted list of unique investors
    investor_list = sorted(set(','.join(df['investors'].dropna()).split(',')))

    # Select investor and update session state
    selected_investor = st.sidebar.selectbox(
        '💸 Select Investor', investor_list
    )

    if st.sidebar.button('📈 Show Investor Details'):
        st.session_state.selected_investor = selected_investor

    # Load investor details if selected
    if st.session_state.selected_investor:
        load_investor_analysis(df, st.session_state.selected_investor)

# Handle Investor Analysis
elif option == 'Investor':
    # Initialize session state for investor selection if not already set
    if 'selected_investor' not in st.session_state:
        st.session_state.selected_investor = None

    # Generate sorted list of unique investors
    investor_list = sorted(set(','.join(df['investors'].dropna()).split(',')))

    # Select investor and update session state
    selected_investor = st.sidebar.selectbox(
        '💸 Select Investor', investor_list
    )

    if st.sidebar.button('📈 Show Investor Details'):
        st.session_state.selected_investor = selected_investor

    # Load investor details if selected
    if st.session_state.selected_investor:
        load_investor_analysis(st.session_state.selected_investor)


# Feedback Section
def feedback_section():
    st.sidebar.markdown('---')
    st.sidebar.subheader('💬 Your Feedback')

    feedback_type = st.sidebar.selectbox(
        '🙋 Select Feedback Type',
        ['General', 'Bug Report', 'Feature Request']
    )

    feedback_text = st.sidebar.text_area(
        '📝 Share your thoughts or suggestions:', ''
    )

    if st.sidebar.button('📤 Submit Feedback'):
        if feedback_text.strip():
            st.sidebar.success('✅ Thank you for your feedback!')
            # Optionally, log or save feedback to a database/file
        else:
            st.sidebar.error('⚠️ Feedback cannot be empty. Please enter something.')


# Call feedback section at the end
feedback_section()
