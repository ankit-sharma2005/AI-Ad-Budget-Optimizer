import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import generate_data_ad, train_model

# page configure
st.set_page_config(
    page_title = "AI Ad Budget Optimizer",
    page_icon = "💻",
    layout = "wide"
)
st.markdown("""
            <style>
            .main {
            background-color:#f5f7fa;}
            .block-container {
            padding-top:1rem;}
            .metric-card {
            background-color : white;
            padding: 15px;
            border-radius: 15px;
            }
            .insight-box {
            background-color: #ecfdf5;
            padding: 20px;
            border-radius:12px;
            border-left : 8px solid green;
            margin-top:10px;
            }
            .warning-box {
            background-color:#fff7ed;
            padding: 20px;
            border-radius: 12px;
            border-left: 8px solid orange;
            margin-top:10px;
            }
            </style>
            """, unsafe_allow_html = True)
st.title("🎯 AI Ad Budget Optimizer")
st.markdown(""" 
            ### Developed by Ankit Sharma 🚀""")
st.sidebar.header('⚙️ Configuration')
show_data = st.sidebar.checkbox('Show Raw Dataset')

alpha = st.sidebar.slider(
    "Regularization Strength (Alpha)",
    min_value = 0.1,
    max_value = 20.0,
    value = 1.0
    )
df = generate_data_ad()
if show_data:
    st.subheader('Raw Marketing Dataset')
    st.dataframe(df)
lasso_model,ridge_model,x_test,y_test,features=train_model(df,alpha)
lasso_coefs = pd.DataFrame({
    'Channel':features,
    'Weight':lasso_model.coef_
    })
ridge_coefs = pd.DataFrame({
    'Channel':features,
    'Weight': ridge_model.coef_
})
best_channel = lasso_coefs.loc[lasso_coefs['Weight'].idxmax()]['Channel']

weak_Channel = lasso_coefs.loc[
    lasso_coefs['Weight'].idxmin()
    ]['Channel']
    
active_features = len(
    lasso_coefs[lasso_coefs['Weight'] != 0]
    )
tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "💰 Revenue Simulator",
        "🧠 Model Analysis",
        "📈 Visualization",
        "🤖 AI Insights"
])

with tab1:
    st.subheader('Marketing Intelligence Dashboard')
    st.markdown("""
                Welcome to the AI Ad budget optimizer.
                This system helps businesses:
                - Analyze advertisement channels
                - predict revenue from marketing spend
                - Identify weak campaigns
                - Optimize ad investment using Machine Learning
                """)
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.metric('Best Channel',best_channel)
    with k2:
        st.metric('weakest channel',weak_Channel)
    with k3:
        st.metric('Active Features',active_features)
    with k4:
        st.metric("Alpha",alpha)
    st.divider()

    st.subheader('Dataset Reviews')
    d1,d2,d3 = st.columns(3)
    with d1:
        st.metric('Total Records',df.shape[0])
    with d2:
        st.metric('Total Features',len(features))
    with d3:
        st.metric("Target Variable","Revenue")  
with tab2:
    st.subheader("User Input Revenue")
    st.markdown("""
                Enter advertisement budgets and let AI predict:
                ✅ Expected Revenue
                ✅ Estimated Profit
                ✅ Marketing ROI
                """)
    scenario = st.selectbox(
        "Choose Marketing Strategy",
        [
            "Custom",
            "Conservative Marketing",
            "Aggressive Campaign",
            "Social Media Focus"
        
        ]
    )
    default_budget = {
        "Custom": 500,
        "Conservative Marketing": 200,
        "Aggressive Campaign":1000,
        "Social Media Focus": 700

    }
    input_data = {}
    cols = st.columns(2)
    for i, feat in enumerate(features):
        with cols[i % 2]:
            input_data[feat] = st.number_input(
                f"Budget for {feat}",
                value = default_budget[scenario]
            )
    if st.button("predict Business Revenue"):
        input_df = pd.DataFrame([input_data])
        lasso_pred = lasso_model.predict(input_df)[0]
        ridge_pred = ridge_model.predict(input_df)[0]
        avg_prediction = (lasso_pred + ridge_pred)/2
        total_spend = sum(input_data.values())
        estimated_profit = avg_prediction - total_spend
        if total_spend > 0:
            roi = (estimated_profit / total_spend) * 100
        else:
            roi = 0

        st.divider()
        st.subheader('Campaign Result')
        r1,r2,r3,r4 = st.columns(4)
        with r1:
            st.metric('✅ Total Ad Spend',f'Rs. {total_spend:,.2f}')
        with r2:
                st.metric("⚡Predicted Revenue",f"Rs. {avg_prediction:.2f}")
        with r3:
                st.metric("💰 Estimated Profit",f"Rs.    {estimated_profit:,.2f}")
        with r4:
            st.metric("🎯 ROI", f"{roi:.2f}%")

        st.success(
            f"Based on your advertisement strategy, "
            f"the AI system predicts an estimated revenue "
            f"of Rs. {avg_prediction:,.2f}"
)
with tab3:
    st.subheader('Machine learning Model Analysis')
    st.markdown("""This section explains how Lasso and Ridge Regression analyze advertisement channels.""")
    col1, col2 = st.columns(2)
    with col1:
        st.caption('Feature Selection Model')
        st.info("""Lasso removes weak advertisement channels by shrinking their coefficient toward zero.""")
        st.dataframe(lasso_coefs)
        active = lasso_coefs[lasso_coefs['Weight'] != 0]
        eliminated = lasso_coefs[lasso_coefs['Weight'] == 0]
        st.success(f"Active Channels: {', '.join(active['Channel'].tolist())}")
        if len(eliminated) > 0:
            st.error(f"Eliminated Channels: {', '.join(eliminated['Channel'].tolist())}")
        else:
            st.warning("No channels eliminated at current Alpha.")
    with col2:
        st.subheader("Ridge Regression")
        st.caption("Coefficient Stabilization Model")
        st.info("""Ridge keeps all features but reduces extreme coefficient values to prevent overfitting.""")
        st.dataframe(ridge_coefs)
    st.divider()
    st.subheader("📈 Regularization Comparison")
    comparison_df = pd.DataFrame({
        "Channel" : features,
        "Lasso" : lasso_model.coef_,
        "Ridge" : ridge_model.coef_
    })
    st.dataframe(comparison_df)

with tab4:
    st.subheader("Feature Impact Visualization")
    st.markdown("""These visualizations help understand how advertisement channels influence revenue.""")
    
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        features,
        lasso_model.coef_,
        marker = 'o',
        linewidth = 3,
        label = 'Lasso'

    )
    ax.plot(
        features,
        ridge_model.coef_,
        marker = 's',
        linewidth = 3,
        label = 'Ridge'
    )
    ax.axhline(0,linestyle ='--')
    ax.set_title("Lasso vs Ridge Coefficient Comparison")
    ax.set_ylabel("Impact Weight")
    ax.set_xlabel("Advertisement Channels")
    ax.legend()
    st.pyplot(fig)
    st.divider()
    st.subheader("📊 Lasso Feature Importance")
    fig2, ax2 = plt.subplots(figsize =(10,5))
    ax2.barh(
        features,
        lasso_model.coef_
    )
    ax2.set_title("Advertisement Channel Impact")
    ax2.set_xlabel("Coefficient Weight")
    st.pyplot(fig2)
with tab5:
    st.subheader("AI Business Recommendations")
    st.markdown("""The AI engine analyzes advertisement performance and suggests business improvements.""")
    positive_channels = lasso_coefs[
        lasso_coefs['Weight'] >0 ]['Channel'].tolist()
    negative_channels = lasso_coefs[
        lasso_coefs['Weight'] < 0 ]['Channel'].tolist()
    
    st.markdown(f"""
                <div class ="insight-box">
                <h4>📈 Growth Recommendations</h4>
                <ul>
                <li>Increase investment in: <b>{', '.join(positive_channels)}</b></li>
                <li>These channels positively influence revenue generation.</li>
                </ul>
                </div>""",unsafe_allow_html = True)
    if len(negative_channels) > 0:

        st.markdown(f"""
            <div class="warning-box">

            <h4>⚠️Budget Optimization Warning</h4>

            <ul>

            <li>
            Consider reducing spend on:
            <b>{', '.join(negative_channels)}</b>
            </li>

            <li>
            These channels show negative contribution trends.
            </li>

            </ul>

            </div>
            """,
            unsafe_allow_html=True
        )