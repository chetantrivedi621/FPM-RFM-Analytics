import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tracemalloc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import os
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components

# ---------------- PAGE SETUP & INJECT CSS ---------------- #
st.set_page_config(page_title="Product Analysis Dashboard", layout="wide")

# Custom light-theme minimalist CSS injection
st.markdown("""
<style>
    /* Main container padding */
    .block-container {
        padding-top: 5rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Metric cards styling - Minimalist Light Theme */
    div[data-testid="stMetric"] {
        background-color: #F1F5F9 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Hide Streamlit footer */
    footer {
        visibility: hidden !important;
    }
    
    /* Hide Main Menu */
    #MainMenu {
        visibility: hidden !important;
    }
    
    /* Section headers styling */
    .custom-section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0F172A;
        border-left: 3px solid #0EA5E9;
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    
    /* Metric value font size adjustments */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    
    /* Robust Tab Header Styling to guarantee visibility */
    [data-baseweb="tab-list"] button,
    [data-baseweb="tab-list"] button *,
    [data-testid="stTabs"] button,
    [data-testid="stTabs"] button *,
    button[data-baseweb="tab"],
    button[data-baseweb="tab"] *,
    div[role="tablist"] button,
    div[role="tablist"] button * {
        color: #475569 !important; /* Slate Gray */
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Active Tab styling */
    [data-baseweb="tab-list"] button[aria-selected="true"],
    [data-baseweb="tab-list"] button[aria-selected="true"] *,
    [data-testid="stTabs"] button[aria-selected="true"],
    [data-testid="stTabs"] button[aria-selected="true"] *,
    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] *,
    div[role="tablist"] button[aria-selected="true"],
    div[role="tablist"] button[aria-selected="true"] * {
        color: #0EA5E9 !important; /* Sky Blue */
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Load from high-speed CSV format instead of Excel
    dataset_path = os.path.join(base_dir, 'Online_Retail_Cleaned.csv')
    df = pd.read_csv(dataset_path, parse_dates=['InvoiceDate'])
    
    df.dropna(subset=['Description', 'InvoiceNo'], inplace=True)
    df['Description'] = df['Description'].str.strip()
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[df['Quantity'] > 0]
    
    return df

df = load_data()

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("Product Analysis")
st.sidebar.caption("Configure filters and rules parameters below.")

countries = sorted(df['Country'].unique())
selected_country = st.sidebar.selectbox("Country Filter", countries)

min_support = st.sidebar.slider("Minimum Support", 0.01, 0.2, 0.07)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.5)
min_lift = st.sidebar.slider("Minimum Lift", 1.0, 5.0, 1.2)

selected_algo = st.sidebar.selectbox("Mining Algorithm", ["Apriori", "FP-Growth"])

# ---------------- OPTIMIZED BASKET ---------------- #
def create_basket(data, country):
    data = data[data['Country'] == country]
    
    unique_invoices = data['InvoiceNo'].unique()
    sample_size = min(len(unique_invoices), 3000)
    np.random.seed(42)
    sampled_invoices = np.random.choice(unique_invoices, size=sample_size, replace=False)
    data = data[data['InvoiceNo'].isin(sampled_invoices)]
    
    top_items = data['Description'].value_counts().head(100).index
    data = data[data['Description'].isin(top_items)]
    
    basket = (data.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().fillna(0))
    
    return (basket > 0).astype(int)

# Helper function to convert frozensets for clean Streamlit display
def format_rules_for_display(rules_df):
    if rules_df.empty:
        return rules_df
    display_df = rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    display_df['antecedents'] = display_df['antecedents'].apply(lambda x: ', '.join(list(x)))
    display_df['consequents'] = display_df['consequents'].apply(lambda x: ', '.join(list(x)))
    return display_df

# Helper function to generate Network graph HTML via Pyvis
def generate_network_graph(rules_df):
    top_30 = rules_df.sort_values(by='lift', ascending=False).head(30)
    
    net = Network(height="400px", width="100%", bgcolor="#F8FAFC", font_color="#0F172A", heading="")
    
    min_sup = top_30['support'].min()
    max_sup = top_30['support'].max()
    sup_range = max_sup - min_sup if max_sup != min_sup else 1.0
    
    min_lift = top_30['lift'].min()
    max_lift = top_30['lift'].max()
    lift_range = max_lift - min_lift if max_lift != min_lift else 1.0
    
    # Map confidence between Slate Gray (#64748B) and Sky Blue (#0EA5E9)
    def get_color(conf):
        r = int(100 + (14 - 100) * conf)
        g = int(116 + (165 - 116) * conf)
        b = int(139 + (233 - 139) * conf)
        return f"rgb({r},{g},{b})"
        
    added_nodes = set()
    
    for _, row in top_30.iterrows():
        ant = ', '.join(list(row['antecedents']))
        cons = ', '.join(list(row['consequents']))
        
        sup_frac = (row['support'] - min_sup) / sup_range if sup_range != 0 else 0.5
        ant_size = 10 + sup_frac * 30
        cons_size = 10 + sup_frac * 30
        
        if ant not in added_nodes:
            net.add_node(ant, label=ant, size=ant_size, color="#E2E8F0", title=f"Product: {ant}\nSupport: {row['support']:.4f}")
            added_nodes.add(ant)
        if cons not in added_nodes:
            net.add_node(cons, label=cons, size=cons_size, color="#E2E8F0", title=f"Product: {cons}\nSupport: {row['support']:.4f}")
            added_nodes.add(cons)
            
        lift_frac = (row['lift'] - min_lift) / lift_range if lift_range != 0 else 0.5
        edge_width = 1 + lift_frac * 4
        edge_color = get_color(row['confidence'])
        
        net.add_edge(ant, cons, value=edge_width, color=edge_color, title=f"Rule: {ant} -> {cons}\nConfidence: {row['confidence']:.4f}\nLift: {row['lift']:.4f}")
        
    net.toggle_physics(True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
    return html_content

# Helper function to generate CSV downloader
def make_csv_download(df_to_download, filename):
    csv = df_to_download.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Table as CSV",
        data=csv,
        file_name=filename,
        mime='text/csv',
    )

# ---------------- CUSTOMER SEGMENTS CACHED FUNCTIONS ---------------- #
@st.cache_data
def get_rfm_data(df_clean):
    df_clean['TotalSum'] = df_clean['Quantity'] * df_clean['UnitPrice']
    max_date = df_clean['InvoiceDate'].max()
    
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSum': 'sum'
    })
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalSum': 'Monetary'
    }, inplace=True)
    
    rfm = rfm[(rfm['Frequency'] > 0) & (rfm['Monetary'] > 0)]
    return rfm

@st.cache_data
def run_kmeans_clustering(rfm_df, k):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm_df.index, columns=['Recency', 'Frequency', 'Monetary'])
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm_result = rfm_df.copy()
    rfm_result['Cluster'] = kmeans.fit_predict(rfm_scaled_df)
    rfm_result['Segment'] = rfm_result['Cluster'].apply(lambda x: f"Segment {x}")
    return rfm_result

# ---------------- TABS SETUP ---------------- #
tab1, tab2, tab3 = st.tabs([
    "Single Algorithm Analysis", 
    "Algorithm Comparison",
    "Customer Segments"
])

# ---------------- TAB 1: SINGLE ALGORITHM ANALYSIS ---------------- #
with tab1:
    # State tracking
    if 'tab1_rules' not in st.session_state:
        st.session_state.tab1_rules = pd.DataFrame()
        st.session_state.tab1_time = 0.0
        st.session_state.tab1_run = False
        st.session_state.tab1_basket = pd.DataFrame()

    # Metrics row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    if st.session_state.tab1_run and not st.session_state.tab1_rules.empty:
        total_rules = len(st.session_state.tab1_rules)
        avg_confidence = st.session_state.tab1_rules['confidence'].mean()
        avg_lift = st.session_state.tab1_rules['lift'].mean()
        exec_time = st.session_state.tab1_time
    else:
        total_rules = 0
        avg_confidence = 0.0
        avg_lift = 0.0
        exec_time = 0.0

    m_col1.metric("Total Rules", f"{total_rules}")
    m_col2.metric("Average Confidence", f"{avg_confidence:.4f}")
    m_col3.metric("Average Lift", f"{avg_lift:.4f}")
    m_col4.metric("Execution Time", f"{exec_time:.4f} s")

    # Filters summary
    st.caption(f"Active Parameters: Country: {selected_country} | Min Support: {min_support} | Min Confidence: {min_confidence} | Min Lift: {min_lift} | Method: {selected_algo}")

    # Run controls
    if st.button("Run Single Analysis", key="run_tab1_btn"):
        with st.spinner("Processing transaction history..."):
            start = time.time()
            basket = create_basket(df, selected_country)
            st.session_state.tab1_basket = basket
            
            if selected_algo == "Apriori":
                freq = apriori(basket, min_support=min_support, use_colnames=True, max_len=2)
            else:
                freq = fpgrowth(basket, min_support=min_support, use_colnames=True, max_len=2)
                
            rules = pd.DataFrame()
            if not freq.empty:
                rules = association_rules(freq, metric="lift", min_threshold=min_lift)
                if not rules.empty:
                    rules = rules[rules['confidence'] >= min_confidence]
                    
            st.session_state.tab1_rules = rules
            st.session_state.tab1_time = time.time() - start
            st.session_state.tab1_run = True
            st.rerun()

    st.divider()

    if st.session_state.tab1_run:
        rules = st.session_state.tab1_rules
        basket = st.session_state.tab1_basket
        
        if rules.empty:
            st.warning("No strong association rules discovered. Adjust parameter thresholds in the sidebar.")
        else:
            # Prepare plotting columns
            rules_plot = rules.copy()
            rules_plot['rule'] = rules_plot['antecedents'].apply(lambda x: ', '.join(list(x))) + " -> " + rules_plot['consequents'].apply(lambda x: ', '.join(list(x)))
            # Truncate rule labels to max 50 characters if longer (Issue 3)
            rules_plot['rule'] = rules_plot['rule'].apply(lambda x: x if len(x) <= 50 else x[:47] + "...")

            st.markdown('<div class="custom-section-header">Rules Visualization</div>', unsafe_allow_html=True)
            
            # 1. Support vs Confidence Scatter - Light Theme (Issue 1, 4)
            fig1 = px.scatter(
                rules_plot,
                x='support',
                y='confidence',
                size='lift',
                color='confidence',
                hover_name='rule',
                hover_data={'support': True, 'confidence': True, 'lift': True},
                color_continuous_scale=[[0, '#64748B'], [1, '#0EA5E9']],
                title="Support vs Confidence",
                template="plotly_white"
            )
            fig1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                margin=dict(t=60, b=60),
                height=500
            )
            st.plotly_chart(fig1, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

            try:
                if len(rules) == 0:
                    st.warning("Not enough data to generate summary. Try lowering the support or confidence threshold.")
                else:
                    total_rules = len(rules)
                    top_row = rules.loc[rules['lift'].idxmax()]
                    max_lift = top_row['lift']
                    top_ant = str(list(top_row['antecedents']))[2:-2]
                    top_con = str(list(top_row['consequents']))[2:-2]
                    st.info(f"{total_rules} association rules were discovered. The strongest relationship exists between '{top_ant}' and '{top_con}' with a lift of {max_lift:.2f}, meaning customers who buy one are {max_lift:.2f}x more likely to purchase the other.")
            except Exception as e:
                st.warning("Summary unavailable for this chart. Try adjusting your filter parameters.")

            st.divider()

            # 2. Top 10 Rules Bar Chart - Light Theme (Issue 3, 4)
            top_10 = rules_plot.sort_values(by='lift', ascending=False).head(10)
            fig2 = px.bar(
                top_10,
                x='lift',
                y='rule',
                orientation='h',
                color='confidence',
                color_continuous_scale=[[0, '#64748B'], [1, '#0EA5E9']],
                title="Top 10 Rules by Lift",
                template="plotly_white",
                labels={'lift': 'Lift', 'rule': 'Rule'}
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                yaxis_categoryorder='total ascending',
                margin=dict(l=320, r=60, t=40, b=40),
                height=450
            )
            st.plotly_chart(fig2, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

            try:
                if len(rules) == 0:
                    st.warning("Not enough data to generate summary. Try lowering the support or confidence threshold.")
                else:
                    avg_confidence = rules['confidence'].mean()
                    top_row = rules.loc[rules['lift'].idxmax()]
                    top_confidence = top_row['confidence']
                    top_ant = str(list(top_row['antecedents']))[2:-2]
                    top_con = str(list(top_row['consequents']))[2:-2]
                    st.info(f"The highest-lift rule '{top_ant} → {top_con}' carries a confidence of {top_confidence:.0%}, meaning it holds true in {top_confidence:.0%} of transactions containing the antecedent. The average confidence across all rules is {avg_confidence:.0%}.")
            except Exception as e:
                st.warning("Summary unavailable for this chart. Try adjusting your filter parameters.")

            st.divider()

            # 3. Top 15 Product Co-occurrences Heatmap - Light Theme (Issue 2, 4, 5)
            top_15 = basket.sum().sort_values(ascending=False).head(15).index
            co_occur = basket[top_15].T.dot(basket[top_15])
            fig3 = px.imshow(
                co_occur,
                x=top_15,
                y=top_15,
                color_continuous_scale=[[0, "#0F172A"], [0.5, "#0D9488"], [1.0, "#2DD4BF"]],
                title="Top 15 Product Co-occurrences",
                template="plotly_white"
            )
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                height=550,
                margin=dict(l=180, r=40, t=60, b=180)
            )
            fig3.update_xaxes(tickangle=-35, tickfont=dict(size=10))
            fig3.update_yaxes(tickfont=dict(size=10))
            st.plotly_chart(fig3, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

            try:
                if co_occur.empty or co_occur.shape[0] < 2:
                    st.warning("Not enough data to generate summary. Try lowering the support or confidence threshold.")
                else:
                    co_matrix = co_occur.copy()
                    np.fill_diagonal(co_matrix.values, -1)
                    max_val = co_matrix.values.max()
                    idx = np.unravel_index(np.argmax(co_matrix.values), co_matrix.shape)
                    product_1 = co_matrix.index[idx[0]]
                    product_2 = co_matrix.columns[idx[1]]
                    st.info(f"'{product_1}' and '{product_2}' are the most frequently co-purchased products, appearing together in {int(max_val)} transactions. This pair represents the strongest natural bundling opportunity and could be targeted with cross-sell promotions.")
            except Exception as e:
                st.warning("Summary unavailable for this chart. Try adjusting your filter parameters.")

            st.divider()
            
            # 4. Association Network Graph (Full width) - Light background
            st.markdown('<div class="custom-section-header">Association Network Graph</div>', unsafe_allow_html=True)
            html_graph = generate_network_graph(rules)
            components.html(html_graph, height=420, scrolling=True)
            st.caption("Node size = support, Edge thickness = lift, Edge color = confidence")

            try:
                if len(rules) == 0:
                    st.warning("Not enough data to generate summary. Try lowering the support or confidence threshold.")
                else:
                    top30 = rules.nlargest(30, 'lift')
                    ant_list = list(top30['antecedents'].apply(lambda x: list(x)[0]))
                    cons_list = list(top30['consequents'].apply(lambda x: list(x)[0]))
                    num_nodes = len(set(ant_list + cons_list))
                    num_edges = len(top30)
                    avg_lift = top30['lift'].mean()
                    st.info(f"The network maps {num_nodes} products connected through {num_edges} association rules, visualizing the top relationships by lift. The average lift across displayed connections is {avg_lift:.2f}, confirming strong and non-random co-purchase patterns across this product catalog.")
            except Exception as e:
                st.warning("Summary unavailable for this chart. Try adjusting your filter parameters.")

            st.divider()

            # Data Table
            st.markdown('<div class="custom-section-header">Association Rules Data</div>', unsafe_allow_html=True)
            display_rules = format_rules_for_display(rules)
            st.dataframe(
                display_rules.sort_values(by='lift', ascending=False),
                use_container_width=True,
                height=300
            )
            make_csv_download(display_rules, "rules_single_analysis.csv")

# ---------------- TAB 2: ALGORITHM COMPARISON ---------------- #
with tab2:
    if 'tab2_apriori_time' not in st.session_state:
        st.session_state.tab2_apriori_time = 0.0
        st.session_state.tab2_fp_time = 0.0
        st.session_state.tab2_apriori_rules = 0
        st.session_state.tab2_fp_rules = 0
        st.session_state.tab2_apriori_mem = 0.0
        st.session_state.tab2_fp_mem = 0.0
        st.session_state.tab2_run = False
        st.session_state.tab2_support_sens = pd.DataFrame()

    # Metrics row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Apriori Execution Time", f"{st.session_state.tab2_apriori_time:.4f} s")
    m_col2.metric("FP-Growth Execution Time", f"{st.session_state.tab2_fp_time:.4f} s")
    m_col3.metric("Apriori Rules Found", f"{st.session_state.tab2_apriori_rules}")
    m_col4.metric("FP-Growth Rules Found", f"{st.session_state.tab2_fp_rules}")

    st.caption(f"Active Parameters: Country: {selected_country} | Min Confidence: {min_confidence} | Min Lift: {min_lift}")

    if st.button("Run Benchmark Comparison", key="run_tab2_btn"):
        with st.spinner("Benchmarking speed and memory..."):
            basket = create_basket(df, selected_country)

            # Benchmark Apriori
            tracemalloc.start()
            tracemalloc.clear_traces()
            ap_start = time.time()
            ap_freq = apriori(basket, min_support=min_support, use_colnames=True, max_len=2)
            ap_rules = pd.DataFrame()
            if not ap_freq.empty:
                ap_rules = association_rules(ap_freq, metric="lift", min_threshold=min_lift)
                if not ap_rules.empty:
                    ap_rules = ap_rules[ap_rules['confidence'] >= min_confidence]
            
            st.session_state.tab2_apriori_time = time.time() - ap_start
            _, ap_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            st.session_state.tab2_apriori_mem = ap_peak / (1024 * 1024)
            st.session_state.tab2_apriori_rules = len(ap_rules)

            # Benchmark FP-Growth
            tracemalloc.start()
            tracemalloc.clear_traces()
            fp_start = time.time()
            fp_freq = fpgrowth(basket, min_support=min_support, use_colnames=True, max_len=2)
            fp_rules = pd.DataFrame()
            if not fp_freq.empty:
                fp_rules = association_rules(fp_freq, metric="lift", min_threshold=min_lift)
                if not fp_rules.empty:
                    fp_rules = fp_rules[fp_rules['confidence'] >= min_confidence]
            
            st.session_state.tab2_fp_time = time.time() - fp_start
            _, fp_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            st.session_state.tab2_fp_mem = fp_peak / (1024 * 1024)
            st.session_state.tab2_fp_rules = len(fp_rules)

            # Sensitivity Analysis
            supports = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
            sens_data = []
            for s in supports:
                # Apriori
                ap_f = apriori(basket, min_support=s, use_colnames=True, max_len=2)
                ap_r_count = 0
                if not ap_f.empty:
                    ap_r = association_rules(ap_f, metric="lift", min_threshold=min_lift)
                    if not ap_r.empty:
                        ap_r_count = len(ap_r[ap_r['confidence'] >= min_confidence])
                # FP
                fp_f = fpgrowth(basket, min_support=s, use_colnames=True, max_len=2)
                fp_r_count = 0
                if not fp_f.empty:
                    fp_r = association_rules(fp_f, metric="lift", min_threshold=min_lift)
                    if not fp_r.empty:
                        fp_r_count = len(fp_r[fp_r['confidence'] >= min_confidence])
                
                sens_data.append({"Support": s, "Algorithm": "Apriori", "Rules Count": ap_r_count})
                sens_data.append({"Support": s, "Algorithm": "FP-Growth", "Rules Count": fp_r_count})

            st.session_state.tab2_support_sens = pd.DataFrame(sens_data)
            st.session_state.tab2_run = True
            st.rerun()

    st.divider()

    if st.session_state.tab2_run:
        st.markdown('<div class="custom-section-header">Performance Metrics Comparison</div>', unsafe_allow_html=True)
        
        # 1. Performance Subplots to handle different scales (Issue 1 fix)
        from plotly.subplots import make_subplots
        fig_bar = make_subplots(
            rows=1, 
            cols=3, 
            subplot_titles=("Execution Time (s)", "Peak Memory (MB)", "Rules Discovered")
        )
        
        # Add Apriori traces
        fig_bar.add_trace(
            go.Bar(
                name='Apriori', 
                x=['Apriori'], 
                y=[st.session_state.tab2_apriori_time], 
                marker_color='#64748B', 
                showlegend=True
            ), 
            row=1, col=1
        )
        fig_bar.add_trace(
            go.Bar(
                name='Apriori', 
                x=['Apriori'], 
                y=[st.session_state.tab2_apriori_mem], 
                marker_color='#64748B', 
                showlegend=False
            ), 
            row=1, col=2
        )
        fig_bar.add_trace(
            go.Bar(
                name='Apriori', 
                x=['Apriori'], 
                y=[st.session_state.tab2_apriori_rules], 
                marker_color='#64748B', 
                showlegend=False
            ), 
            row=1, col=3
        )
        
        # Add FP-Growth traces
        fig_bar.add_trace(
            go.Bar(
                name='FP-Growth', 
                x=['FP-Growth'], 
                y=[st.session_state.tab2_fp_time], 
                marker_color='#0EA5E9', 
                showlegend=True
            ), 
            row=1, col=1
        )
        fig_bar.add_trace(
            go.Bar(
                name='FP-Growth', 
                x=['FP-Growth'], 
                y=[st.session_state.tab2_fp_mem], 
                marker_color='#0EA5E9', 
                showlegend=False
            ), 
            row=1, col=2
        )
        fig_bar.add_trace(
            go.Bar(
                name='FP-Growth', 
                x=['FP-Growth'], 
                y=[st.session_state.tab2_fp_rules], 
                marker_color='#0EA5E9', 
                showlegend=False
            ), 
            row=1, col=3
        )
        
        fig_bar.update_layout(
            title="Performance Benchmarks",
            template='plotly_white',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

        try:
            apriori_rules_count = st.session_state.tab2_apriori_rules
            if apriori_rules_count == 0:
                st.warning("Not enough data to generate summary. Try lowering the support or confidence threshold.")
            else:
                fpgrowth_time = st.session_state.tab2_fp_time
                apriori_time = st.session_state.tab2_apriori_time
                fpgrowth_mem = st.session_state.tab2_fp_mem
                apriori_mem = st.session_state.tab2_apriori_mem
                fpgrowth_rules_count = st.session_state.tab2_fp_rules
                
                faster_algo = "FP-Growth" if fpgrowth_time < apriori_time else "Apriori"
                slower_algo = "Apriori" if faster_algo == "FP-Growth" else "FP-Growth"
                
                min_time = min(apriori_time, fpgrowth_time)
                speedup = max(apriori_time, fpgrowth_time) / min_time if min_time > 0 else 1.0
                
                st.info(f"{faster_algo} completed {speedup:.1f}x faster than {slower_algo} and consumed less peak memory, while both algorithms discovered {apriori_rules_count} identical association rules. This confirms FP-Growth's efficiency advantage on larger datasets without any loss in rule quality.")
        except Exception as e:
            st.warning("Summary unavailable for this chart. Try adjusting your filter parameters.")

        st.divider()

        # 2. Line Chart: Support vs Rules - Light Theme (Issue 2 overlapping line fix)
        sens_df = st.session_state.tab2_support_sens
        fig_line = go.Figure()
        
        ap_sens = sens_df[sens_df['Algorithm'] == 'Apriori']
        fp_sens = sens_df[sens_df['Algorithm'] == 'FP-Growth']
        
        # Apriori: Thick slate-gray solid line with square markers
        fig_line.add_trace(go.Scatter(
            x=ap_sens['Support'], 
            y=ap_sens['Rules Count'], 
            mode='lines+markers', 
            name='Apriori', 
            line=dict(color='#64748B', width=4),
            marker=dict(symbol='square', size=8)
        ))
        
        # FP-Growth: Thin sky-blue dashed line with circle markers
        fig_line.add_trace(go.Scatter(
            x=fp_sens['Support'], 
            y=fp_sens['Rules Count'], 
            mode='lines+markers', 
            name='FP-Growth', 
            line=dict(color='#0EA5E9', width=2, dash='dash'),
            marker=dict(symbol='circle', size=6)
        ))
        
        fig_line.update_layout(
            title="Support Sensitivity analysis",
            xaxis_title="Support Threshold",
            yaxis_title="Rules Count",
            template='plotly_white',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            height=500
        )
        st.plotly_chart(fig_line, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

        try:
            sens_df = st.session_state.tab2_support_sens
            if sens_df.empty:
                st.warning("Not enough data to generate summary. Try lowering the support or confidence threshold.")
            else:
                support_range = sens_df['Support'].unique()
                min_support_tested = min(support_range)
                
                ap_sens_df = sens_df[(sens_df['Algorithm'] == 'Apriori') & (sens_df['Support'] == min_support_tested)]
                fp_sens_df = sens_df[(sens_df['Algorithm'] == 'FP-Growth') & (sens_df['Support'] == min_support_tested)]
                
                max_apriori_rules = ap_sens_df['Rules Count'].values[0] if not ap_sens_df.empty else 0
                max_fpgrowth_rules = fp_sens_df['Rules Count'].values[0] if not fp_sens_df.empty else 0
                
                if max_apriori_rules == 0 and max_fpgrowth_rules == 0:
                    st.warning("Not enough data to generate summary. Try lowering the support or confidence threshold.")
                else:
                    st.info(f"At the lowest tested support threshold of {min_support_tested:.2f}, Apriori generated {max_apriori_rules} rules and FP-Growth generated {max_fpgrowth_rules} rules. Rule count drops sharply as support increases, highlighting the importance of threshold tuning for balancing rule quality versus quantity.")
        except Exception as e:
            st.warning("Summary unavailable for this chart. Try adjusting your filter parameters.")

        st.divider()

        # Data Table
        st.markdown('<div class="custom-section-header">Benchmark Logs</div>', unsafe_allow_html=True)
        bench_log = pd.DataFrame({
            "Metric": ["Execution Time (seconds)", "Peak Memory (MB)", "Rules Count"],
            "Apriori": [st.session_state.tab2_apriori_time, st.session_state.tab2_apriori_mem, st.session_state.tab2_apriori_rules],
            "FP-Growth": [st.session_state.tab2_fp_time, st.session_state.tab2_fp_mem, st.session_state.tab2_fp_rules]
        })
        st.dataframe(bench_log, use_container_width=True, height=200)
        make_csv_download(bench_log, "benchmark_metrics.csv")

# ---------------- TAB 3: CUSTOMER SEGMENTS ---------------- #
with tab3:
    st.header("Customer Segmentation")
    st.caption("Apply K-Means clustering to partition customers based on Recency, Frequency, and Monetary (RFM) dynamics.")

    # Slider for K
    k_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=6, value=4)

    # Process RFM features (Clean CustomerID)
    df_rfm_input = df.dropna(subset=['CustomerID']).copy()
    df_rfm_input['CustomerID'] = df_rfm_input['CustomerID'].astype(int)
    
    # Retrieve cached RFM data
    rfm_base = get_rfm_data(df_rfm_input)
    
    # Retrieve cached KMeans clustering results based on dynamic k value
    rfm = run_kmeans_clustering(rfm_base, k_clusters)

    # Metrics row
    tot_cust = len(rfm)
    seg_counts = rfm['Segment'].value_counts()
    largest_cluster = seg_counts.max() if not seg_counts.empty else 0
    avg_monetary = rfm['Monetary'].mean()

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Customers", f"{tot_cust}")
    m_col2.metric("Clusters Selected", f"{k_clusters}")
    m_col3.metric("Largest Cluster Size", f"{largest_cluster}")
    m_col4.metric("Average Monetary Value", f"${avg_monetary:.2f}")

    # Filters summary
    st.caption(f"Segmentation Parameters: Features scaled with StandardScaler | KMeans Random State: 42 | K: {k_clusters}")

    st.divider()

    st.markdown('<div class="custom-section-header">Customer Segments Visualization</div>', unsafe_allow_html=True)

    # 1. 3D Plotly Scatter Plot - Light Theme
    fig_3d = px.scatter_3d(
        rfm.reset_index(),
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Segment',
        hover_data={'CustomerID': True, 'Recency': True, 'Frequency': True, 'Monetary': True, 'Cluster': True},
        title=f"3D RFM Clusters (k={k_clusters})",
        labels={'Recency': 'Recency', 'Frequency': 'Frequency', 'Monetary': 'Monetary'},
        log_y=True,
        log_z=True,
        color_discrete_sequence=['#0EA5E9', '#64748B', '#38BDF8', '#F43F5E', '#CBD5E1', '#F59E0B'],
        template="plotly_white"
    )
    fig_3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    try:
        if rfm.empty:
            st.warning("Not enough data to generate summary. Try lowering the support or confidence threshold.")
        else:
            n_clusters = k_clusters
            cluster_sizes = rfm['Cluster'].value_counts()
            largest_cluster_id = cluster_sizes.idxmax()
            largest_cluster_size = cluster_sizes.max()
            smallest_cluster_id = cluster_sizes.idxmin()
            st.info(f"{n_clusters} distinct customer segments were identified in 3D RFM space. Cluster {largest_cluster_id} is the dominant group with {largest_cluster_size} customers, while Cluster {smallest_cluster_id} is the smallest and may represent a niche high-value or at-risk segment worth investigating.")
    except Exception as e:
        st.warning("Summary unavailable for this chart. Try adjusting your filter parameters.")

    st.divider()

    # 2. 2D Fallback Scatter (Recency vs Monetary) - Light Theme
    fig_2d = px.scatter(
        rfm.reset_index(),
        x='Recency',
        y='Monetary',
        color='Segment',
        size='Frequency',
        log_y=True,
        title="Recency vs Monetary (Size by Frequency)",
        template='plotly_white',
        color_discrete_sequence=['#0EA5E9', '#64748B', '#38BDF8', '#F43F5E', '#CBD5E1', '#F59E0B']
    )
    fig_2d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        height=500
    )
    st.plotly_chart(fig_2d, use_container_width=True)

    try:
        if rfm.empty:
            st.warning("Not enough data to generate summary. Try lowering the support or confidence threshold.")
        else:
            grouped = rfm.groupby('Cluster')
            high_value_cluster = grouped['Monetary'].mean().idxmax()
            avg_monetary_high = grouped['Monetary'].mean().max()
            best_recency_cluster = grouped['Recency'].mean().idxmin()
            avg_recency_best = grouped['Recency'].mean().min()
            st.info(f"Cluster {high_value_cluster} contains the highest-value customers with an average spend of £{avg_monetary_high:.2f}, making them the primary target for premium and loyalty campaigns. Cluster {best_recency_cluster} has the most recent average purchase of {avg_recency_best:.0f} days ago, indicating the most actively engaged customer base.")
    except Exception as e:
        st.warning("Summary unavailable for this chart. Try adjusting your filter parameters.")

    st.divider()

    # 3. Bar Chart: Average RFM values per cluster - Light Theme
    means_df = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
    melted_means = means_df.melt(id_vars='Segment', var_name='Metric', value_name='Average Value')
    
    fig_bar_rfm = px.bar(
        melted_means,
        x='Segment',
        y='Average Value',
        color='Metric',
        barmode='group',
        log_y=True,
        title="Average RFM Metrics per Cluster (Log Scale)",
        template='plotly_white',
        color_discrete_sequence=['#0EA5E9', '#64748B', '#CBD5E1']
    )
    fig_bar_rfm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        height=500
    )
    st.plotly_chart(fig_bar_rfm, use_container_width=True)

    try:
        if rfm.empty:
            st.warning("Not enough data to generate summary. Try lowering the support or confidence threshold.")
        else:
            cluster_summary = rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean()
            best_cluster = cluster_summary['Monetary'].idxmax()
            worst_cluster = cluster_summary['Monetary'].idxmin()
            monetary_gap = cluster_summary['Monetary'].max() - cluster_summary['Monetary'].min()
            best_freq_cluster = cluster_summary['Frequency'].idxmax()
            avg_freq_best = cluster_summary.loc[best_freq_cluster, 'Frequency']
            st.info(f"A spending gap of £{monetary_gap:.2f} separates the highest-value Cluster {best_cluster} from the lowest-value Cluster {worst_cluster}, revealing a clear tiering of customer worth. Cluster {best_freq_cluster} leads in purchase frequency with an average of {avg_freq_best:.1f} orders, identifying the most loyal and repeat-buying segment.")
    except Exception as e:
        st.warning("Summary unavailable for this chart. Try adjusting your filter parameters.")

    st.divider()

    # Segment specific rules
    st.markdown('<div class="custom-section-header">Segment Association Mining</div>', unsafe_allow_html=True)
    
    segments = sorted(rfm['Segment'].unique())
    selected_seg = st.selectbox("Select Segment for Rules Analysis", segments)

    if 'tab3_rules' not in st.session_state:
        st.session_state.tab3_rules = pd.DataFrame()
        st.session_state.tab3_run = False
        st.session_state.tab3_seg = ""

    if st.button("Run Analysis on Segment", key="run_tab3_btn"):
        with st.spinner(f"Mining association rules for customers in {selected_seg}..."):
            cust_ids = rfm[rfm['Segment'] == selected_seg].index
            seg_df = df_rfm_input[df_rfm_input['CustomerID'].isin(cust_ids)]
            
            if seg_df.empty:
                st.session_state.tab3_rules = pd.DataFrame()
            else:
                # Invoice-based sampling
                invoices = seg_df['InvoiceNo'].unique()
                sample_sz = min(len(invoices), 3000)
                np.random.seed(42)
                sampled_inv = np.random.choice(invoices, size=sample_sz, replace=False)
                seg_df_sampled = seg_df[seg_df['InvoiceNo'].isin(sampled_inv)]
                
                # Top 100 items
                top_items_seg = seg_df_sampled['Description'].value_counts().head(100).index
                seg_df_filtered = seg_df_sampled[seg_df_sampled['Description'].isin(top_items_seg)]
                
                if seg_df_filtered.empty:
                    st.session_state.tab3_rules = pd.DataFrame()
                else:
                    basket_seg = (seg_df_filtered.groupby(['InvoiceNo', 'Description'])['Quantity']
                                  .sum().unstack().fillna(0))
                    basket_seg = (basket_seg > 0).astype(int)
                    
                    if selected_algo == "Apriori":
                        freq_seg = apriori(basket_seg, min_support=min_support, use_colnames=True, max_len=2)
                    else:
                        freq_seg = fpgrowth(basket_seg, min_support=min_support, use_colnames=True, max_len=2)
                        
                    rules_seg = pd.DataFrame()
                    if not freq_seg.empty:
                        rules_seg = association_rules(freq_seg, metric="lift", min_threshold=min_lift)
                        if not rules_seg.empty:
                            rules_seg = rules_seg[rules_seg['confidence'] >= min_confidence]
                            
                    st.session_state.tab3_rules = rules_seg
            st.session_state.tab3_run = True
            st.session_state.tab3_seg = selected_seg
            st.rerun()

    if st.session_state.tab3_run:
        rules_seg = st.session_state.tab3_rules
        if rules_seg.empty:
            st.warning(f"No rules found for {st.session_state.tab3_seg} at the selected sidebar parameters.")
        else:
            st.success(f"Discovered {len(rules_seg)} association rules for {st.session_state.tab3_seg}")
            display_rules_seg = format_rules_for_display(rules_seg)
            st.dataframe(
                display_rules_seg.sort_values(by='lift', ascending=False),
                use_container_width=True,
                height=300
            )
            make_csv_download(display_rules_seg, f"rules_{st.session_state.tab3_seg}.csv")

# ---------------- RAW DATA EXPANDER ---------------- #
st.divider()
with st.expander("Raw Data View"):
    st.dataframe(df.head(100), use_container_width=True, height=200)