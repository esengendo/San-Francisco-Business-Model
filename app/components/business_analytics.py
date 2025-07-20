"""
Business Analytics Components for SF Business Dashboard
Enhanced business intelligence features and data visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path

def load_actual_data():
    """Load actual business data if available"""
    try:
        # Import config to get dynamic paths
        import sys
        from pathlib import Path
        
        # Add utils to path
        current_dir = Path(__file__).parent
        utils_dir = current_dir.parent / "utils"
        sys.path.append(str(utils_dir))
        
        from config import config
        
        # Try to load actual processed data using config paths
        data_paths = [
            Path(config.X_TRAIN_PATH),
            Path(config.Y_TRAIN_PATH),
            Path(config.BASE_DIR) / "processed" / "sf_business" / "sf_business_processed.parquet",
            Path(config.BASE_DIR) / "raw_data" / "sf_business" / "registered_businesses_raw.parquet"
        ]
    except:
        # Fallback to relative paths if config fails
        current_file = Path(__file__).resolve()
        base_dir = current_file.parent.parent.parent
        data_paths = [
            base_dir / "processed" / "final" / "X_train.parquet",
            base_dir / "processed" / "sf_business" / "sf_business_processed.parquet", 
            base_dir / "raw_data" / "sf_business" / "registered_businesses_raw.parquet"
        ]
    
    for path in data_paths:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                st.success(f"✅ Loaded real data: {len(df):,} records from {path.name}")
                return df
            except Exception as e:
                continue
    
    st.info("📊 Using simulated data for demonstration")
    return None

def create_industry_performance_chart(df=None):
    """Create industry performance visualization"""
    if df is not None and 'industry' in df.columns:
        # Use real data
        industry_stats = df.groupby('industry').agg({
            'business_id': 'count',
            'survival_probability': 'mean' if 'survival_probability' in df.columns else lambda x: np.random.uniform(0.4, 0.8, len(x)).mean()
        }).reset_index()
        industry_stats.columns = ['Industry', 'Business_Count', 'Success_Rate']
        industry_stats = industry_stats.head(10)  # Top 10 industries
    else:
        # Use simulated data
        industry_stats = pd.DataFrame({
            'Industry': ['Professional Services', 'Food & Beverage', 'Retail Trade', 
                        'Healthcare', 'Technology', 'Real Estate', 'Construction',
                        'Arts & Entertainment', 'Finance', 'Education'],
            'Success_Rate': [0.78, 0.58, 0.62, 0.81, 0.74, 0.69, 0.71, 0.45, 0.83, 0.76],
            'Business_Count': [15420, 8930, 12150, 6780, 9240, 5630, 4890, 3420, 2890, 3670]
        })
    
    fig = px.scatter(industry_stats, 
                    x='Business_Count', 
                    y='Success_Rate',
                    size='Business_Count',
                    color='Success_Rate',
                    hover_name='Industry',
                    title='Industry Performance: Business Volume vs Success Rate',
                    labels={'Business_Count': 'Number of Businesses', 
                           'Success_Rate': '5-Year Success Rate'},
                    color_continuous_scale='RdYlGn')
    
    fig.update_layout(height=500)
    return fig

def create_geographic_heatmap():
    """Create geographic business density heatmap"""
    # SF neighborhood data (simplified)
    neighborhoods = {
        'Neighborhood': ['Financial District', 'SOMA', 'Mission', 'Castro', 'Marina', 
                        'Richmond', 'Sunset', 'Chinatown', 'North Beach', 'Haight'],
        'Latitude': [37.7918, 37.7749, 37.7599, 37.7609, 37.8021,
                    37.7799, 37.7449, 37.7941, 37.8066, 37.7699],
        'Longitude': [-122.3992, -122.4194, -122.4148, -122.4350, -122.4364,
                     -122.4782, -122.4691, -122.4078, -122.4103, -122.4493],
        'Business_Density': [95, 88, 72, 65, 70, 45, 40, 68, 58, 52],
        'Success_Rate': [0.84, 0.68, 0.72, 0.81, 0.75, 0.63, 0.59, 0.77, 0.71, 0.66]
    }
    
    df_geo = pd.DataFrame(neighborhoods)
    
    fig = px.scatter_mapbox(df_geo,
                           lat='Latitude',
                           lon='Longitude', 
                           size='Business_Density',
                           color='Success_Rate',
                           hover_name='Neighborhood',
                           hover_data=['Business_Density', 'Success_Rate'],
                           color_continuous_scale='RdYlGn',
                           title='SF Business Success Rate by Neighborhood',
                           mapbox_style='open-street-map',
                           height=500,
                           zoom=11)
    
    fig.update_layout(mapbox_center={"lat": 37.7749, "lon": -122.4194})
    return fig

def create_temporal_trends_chart():
    """Create business trends over time"""
    # Simulated temporal data
    years = list(range(2013, 2025))
    
    # Generate realistic business trends
    np.random.seed(42)
    business_formations = [12000 + i*500 + np.random.randint(-1000, 1000) for i in range(len(years))]
    # COVID impact
    business_formations[7] = 8500  # 2020
    business_formations[8] = 11200  # 2021
    
    success_rates = [0.67 + 0.02*np.sin(i/2) + np.random.uniform(-0.03, 0.03) for i in range(len(years))]
    
    df_temporal = pd.DataFrame({
        'Year': years,
        'New_Businesses': business_formations,
        'Success_Rate': success_rates
    })
    
    fig = go.Figure()
    
    # Add business formations
    fig.add_trace(go.Scatter(
        x=df_temporal['Year'],
        y=df_temporal['New_Businesses'],
        mode='lines+markers',
        name='New Business Registrations',
        line=dict(color='#1f77b4', width=3),
        yaxis='y'
    ))
    
    # Add success rate on secondary axis
    fig.add_trace(go.Scatter(
        x=df_temporal['Year'],
        y=df_temporal['Success_Rate'],
        mode='lines+markers',
        name='Success Rate',
        line=dict(color='#ff7f0e', width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='SF Business Formation & Success Trends (2013-2024)',
        xaxis_title='Year',
        yaxis=dict(title='New Business Registrations', side='left'),
        yaxis2=dict(title='5-Year Success Rate', overlaying='y', side='right', range=[0.5, 0.8]),
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_success_factors_radar():
    """Create radar chart for business success factors"""
    factors = ['Location Score', 'Market Demand', 'Competition Level', 
               'Economic Climate', 'Regulatory Environment', 'Access to Capital']
    
    # Sample business scores
    business_scores = [85, 72, 45, 78, 68, 62]
    market_average = [67, 65, 58, 70, 72, 55]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=business_scores,
        theta=factors,
        fill='toself',
        name='Your Business Profile',
        line_color='#1f77b4'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=market_average,
        theta=factors,
        fill='toself',
        name='Market Average',
        line_color='#ff7f0e',
        opacity=0.6
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title='Business Success Factor Analysis',
        height=400
    )
    
    return fig

def show_financial_projections():
    """Show financial projections and ROI analysis"""
    st.subheader("💰 **Financial Projections & ROI Analysis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**5-Year Revenue Projection**")
        
        # Financial projection inputs
        initial_investment = st.slider("Initial Investment ($)", 50000, 500000, 150000, step=10000)
        monthly_revenue_yr1 = st.slider("Monthly Revenue Year 1 ($)", 10000, 100000, 35000, step=5000)
        growth_rate = st.slider("Annual Growth Rate (%)", 5, 25, 12, step=1) / 100
        
        # Calculate projections
        years = list(range(1, 6))
        revenues = []
        cumulative_profit = []
        
        for year in years:
            annual_revenue = monthly_revenue_yr1 * 12 * (1 + growth_rate) ** (year - 1)
            revenues.append(annual_revenue)
            
            # Simple profit calculation (assuming 20% margin after expenses)
            annual_profit = annual_revenue * 0.2
            if year == 1:
                cumulative_profit.append(annual_profit - initial_investment)
            else:
                cumulative_profit.append(cumulative_profit[-1] + annual_profit)
        
        # Create projection chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=years, y=revenues, name='Annual Revenue', marker_color='lightblue'))
        fig.add_trace(go.Scatter(x=years, y=cumulative_profit, mode='lines+markers', 
                                name='Cumulative Profit', yaxis='y2', line=dict(color='green', width=3)))
        
        fig.update_layout(
            title='5-Year Financial Projection',
            xaxis_title='Year',
            yaxis_title='Annual Revenue ($)',
            yaxis2=dict(title='Cumulative Profit ($)', overlaying='y', side='right'),
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Investment Metrics**")
        
        # Calculate key metrics
        total_revenue_5yr = sum(revenues)
        total_profit_5yr = cumulative_profit[-1] + initial_investment
        roi_5yr = (total_profit_5yr / initial_investment) * 100
        payback_period = next((i for i, profit in enumerate(cumulative_profit, 1) if profit > 0), None)
        
        # Display metrics
        st.metric("5-Year Total Revenue", f"${total_revenue_5yr:,.0f}")
        st.metric("5-Year ROI", f"{roi_5yr:.1f}%", delta=f"vs 25% target")
        st.metric("Break-even Period", f"{payback_period} years" if payback_period else "5+ years")
        st.metric("Year 5 Annual Revenue", f"${revenues[-1]:,.0f}")
        
        # Risk assessment
        st.markdown("**📊 Risk Assessment**")
        if roi_5yr > 100:
            st.success("🟢 **Low Risk**: Strong ROI projection")
        elif roi_5yr > 50:
            st.warning("🟡 **Medium Risk**: Moderate returns expected")
        else:
            st.error("🔴 **High Risk**: Below-target returns")
        
        # Sensitivity analysis
        st.markdown("**🔄 Sensitivity Analysis**")
        scenarios = {
            "Best Case (+20%)": revenues[-1] * 1.2,
            "Base Case": revenues[-1],
            "Worst Case (-30%)": revenues[-1] * 0.7
        }
        
        for scenario, value in scenarios.items():
            st.markdown(f"**{scenario}**: ${value:,.0f}")

def create_competitive_analysis():
    """Create competitive landscape analysis"""
    st.subheader("🏆 **Competitive Landscape Analysis**")
    
    # Competitive data
    competitors = {
        'Business_Type': ['Direct Competitor A', 'Direct Competitor B', 'Indirect Competitor C', 
                         'Your Business', 'Market Leader', 'New Entrant'],
        'Market_Share': [15, 12, 8, 5, 25, 3],
        'Customer_Rating': [4.2, 3.8, 4.1, 4.5, 4.6, 3.9],
        'Price_Point': [85, 78, 65, 80, 95, 60],
        'Years_Operating': [8, 12, 5, 0, 15, 1]
    }
    
    df_comp = pd.DataFrame(competitors)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market share pie chart
        fig1 = px.pie(df_comp, values='Market_Share', names='Business_Type',
                     title='Market Share Distribution',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Competitive positioning
        fig2 = px.scatter(df_comp, x='Price_Point', y='Customer_Rating',
                         size='Market_Share', color='Years_Operating',
                         hover_name='Business_Type',
                         title='Competitive Positioning: Price vs Quality',
                         labels={'Price_Point': 'Price Index', 'Customer_Rating': 'Customer Rating (1-5)'})
        
        # Highlight your business
        your_business = df_comp[df_comp['Business_Type'] == 'Your Business']
        fig2.add_trace(go.Scatter(
            x=your_business['Price_Point'],
            y=your_business['Customer_Rating'],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star'),
            name='Your Position',
            showlegend=True
        ))
        
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Competitive insights
    st.markdown("**🎯 Strategic Positioning Insights:**")
    insights = [
        "**Price Advantage**: Positioned competitively in mid-market segment",
        "**Quality Opportunity**: Target 4.5+ rating to compete with market leader", 
        "**Market Gap**: Underserved premium segment with 20%+ market opportunity",
        "**Differentiation**: Focus on unique value proposition vs established players"
    ]
    
    for insight in insights:
        st.markdown(f"• {insight}")