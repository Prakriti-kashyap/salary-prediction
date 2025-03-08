import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set Streamlit page config
st.set_page_config(layout='wide', page_title='Startup Funding Analysis')


# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv('startup_cleaned.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df


df = load_data()


def load_overall_analysis():
    st.title('Overall Analysis')

    # Compute key metrics
    total_funding = df['amount'].sum()
    max_funding = df.groupby('startup')['amount'].max().max()
    avg_funding = df.groupby('startup')['amount'].sum().mean()
    num_startups = df['startup'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Funding', f"{total_funding:.2f} Cr")
    col2.metric('Max Funding', f"{max_funding:.2f} Cr")
    col3.metric('Avg Ticket Size', f"{avg_funding:.2f} Cr")
    col4.metric('Funded Startups', num_startups)

    # Month-over-Month (MoM) analysis
    st.header('Month-over-Month (MoM) Analysis')
    selected_option = st.selectbox('Select Type', ['Total Investment', 'Number of Investments'])

    if selected_option == 'Total Investment':
        temp_df = df.groupby(['year', 'month'])['amount'].sum().reset_index()
    else:
        temp_df = df.groupby(['year', 'month'])['amount'].count().reset_index()

    temp_df['x_axis'] = temp_df['month'].astype(str) + '-' + temp_df['year'].astype(str)

    # Plotting inside function
    fig, ax = plt.subplots(figsize=(14, 6))  # Increase figure size
    ax.plot(temp_df['x_axis'], temp_df['amount'], marker='o', linestyle='-', label="Funding Amount")

    # Reduce the number of x-ticks dynamically
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.tight_layout()

    # Display plot in Streamlit
    st.pyplot(fig)


def load_investor_details(investor):
    st.title(f'Investor Analysis: {investor}')
    investor_df = df[df['investors'].str.contains(investor, na=False, case=False)]

    # Display recent investments
    st.subheader('Most Recent Investments')
    st.dataframe(investor_df[['date', 'startup', 'vertical', 'city', 'round', 'amount']].head())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Top Investments')
        big_investments = investor_df.groupby('startup')['amount'].sum().nlargest(5)
        fig, ax = plt.subplots()
        ax.bar(big_investments.index, big_investments.values)
        st.pyplot(fig)

    with col2:
        st.subheader('Sectors Invested In')
        sector_distribution = investor_df.groupby('vertical')['amount'].sum()
        if len(sector_distribution) > 1:  # Ensure at least two sectors for pie chart
            fig, ax = plt.subplots()
            ax.pie(sector_distribution, labels=sector_distribution.index, autopct='%0.1f%%')
            st.pyplot(fig)
        else:
            st.write("Only one sector found, skipping pie chart.")

    # Year-over-Year (YoY) Investment
    st.subheader('Year-over-Year Investment Trend')
    yoy_investments = investor_df.groupby('year')['amount'].sum()
    fig, ax = plt.subplots()
    ax.plot(yoy_investments.index, yoy_investments.values, marker='o')
    st.pyplot(fig)


# Sidebar Navigation
st.sidebar.title('Startup Funding Analysis')
option = st.sidebar.selectbox('Select Analysis Type', ['Overall Analysis', 'Startup', 'Investor'])

# Store selection in session state
if 'selected_startup' not in st.session_state:
    st.session_state.selected_startup = None

if 'selected_investor' not in st.session_state:
    st.session_state.selected_investor = None

if option == 'Overall Analysis':
    load_overall_analysis()

elif option == 'Startup':
    selected_startup = st.sidebar.selectbox('Select Startup', sorted(df['startup'].dropna().unique()))
    if st.sidebar.button('Find Startup Details'):
        st.session_state.selected_startup = selected_startup  # Store selection persistently

    if st.session_state.selected_startup:
        st.title(f'Startup Analysis: {st.session_state.selected_startup}')
        st.dataframe(df[df['startup'] == st.session_state.selected_startup])

elif option == 'Investor':
    # Fix investor selection (handle comma-separated names properly)
    investor_list = sorted(set(', '.join(df['investors'].dropna()).split(', ')))  # Fix separator
    selected_investor = st.sidebar.selectbox('Select Investor', investor_list)

    if st.sidebar.button('Find Investor Details'):
        st.session_state.selected_investor = selected_investor  # Store selection persistently

    if st.session_state.selected_investor:
        load_investor_details(st.session_state.selected_investor)
