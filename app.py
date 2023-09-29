import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LogisticRegression
import joblib
from io import StringIO
import streamlit_ext as ste
import base64
from PIL import Image, ImageOps


st.set_page_config(
        page_title="Churn Prediction App",
        page_icon=":chart_with_upwards_trend",
        layout="wide"
    )



page_bg_img = f"""

<style>
[data-testid="stHeader"]{{
    background-color:rgba(0,0,0,0);
}}


[class="css-6qob1r eczjsme3"]{{
    font-family:sans-serif;
    background-color:#00afd6 ;
    color:white;
}}

[data-testid="baseButton-secondary"]{{
    font-family:sans-serif;
    background-color:white;
    color:#00afd6
}}

[class="css-7ym5gk ef3psqc11"]{{
    font-family:sans-serif;
    background-color:#00afd6;
    color: white

}}

[class="css-10trblm e1nzilvr1"]{{
    font-family:sans-serif;
    color: #00afd6;
    font-weight: normal;

}}

[class="css-ompuv2 eqpbllx1"]{{
    font-family:sans-serif;
    color: #00afd6;
    font-weight: bold;
}}

[class="streamlit-expanderHeader st-ae st-cm st-ag st-ah st-ai st-aj st-cn st-co st-cp st-cq st-cr st-cs st-ct st-ar st-as st-b6 st-b5 st-b3 st-cu st-cv st-cw st-b4 st-cx st-cy st-cz st-d0 st-d1"]{{
    background-color:#dcdcdc;
}}

img{{
    max-width:10%;
    max-height: 100%;
    margin-left: auto;
    margin-right: auto;


}}

[data-testid="block-container"]{{
    padding-top:30px
}}

[class="g-gtitle"]{{
    color: #00afd6;
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
# st.title("Churn Prediction App")

st.image("churn-rate2.png", use_column_width=True)

st.markdown("""<p style="font-family:Helvetica;color:#00afd6; font-size: 70px; text-align: center; font-weight:bold;"> Churn Prediction app</p>
            <p style="font-family:sans-serif;color:#00afd6; font-size: 24px;"> Upload a csv file here</p>

            """,unsafe_allow_html=True )


package = joblib.load("model.pkl")
loaded_model = package.named_steps['classifier']
loaded_preproc = package.named_steps['preprocessor']

def main():


    # upload CSV file
    uploaded_file = st.file_uploader("File must follow our template guidelines", type=["csv"])

    st.sidebar.markdown("""
    <p style= "font-family:sans-serif;color:white; font-size: 24px;"> USER INPUT:</p>
    <br/>
    <p style= "font-family:sans-serif;color:white; font-size: 24px;"> Instructions:</p>


    1. Upload a CSV file containing the data you want to predict.
    2. The file should have the same columns as the training data.
    3. After uploading, click the 'Predict' button to see predictions.

    <br/>

    <p style= "font-family:sans-serif;color:white; font-size: 24px;"> Download template here</p>
    """, unsafe_allow_html=True)


    template_data = pd.DataFrame({
    'msno': [0.0],
    'payment_plan_days': [0.0],
    "payment_plan_days": [0.0],
    "plan_list_price": [0.0],
    "actual_amount_paid": [0.0],
    "is_auto_renew": [0.0],
    "is_cancel": [0.0],
    "remaining_plan_duration": [0.0],
    "is_discount": [0.0],
    "num_25": [0.0],
    "num_50": [0.0],
    "num_75": [0.0],
    "num_985": [0.0],
    "num_100": [0.0],
    "num_unq": [0.0],
    "total_secs": [0.0],
    "period_0_churn": [0.0],
    "period_-1_churn": [0.0],
    "period_-2_churn": [0.0],
    "period_-3_churn": [0.0],
    "period_-4_churn": [0.0],
    "period_-5_churn": [0.0],
    "usage_from_ltd": [0.0],
    "discount_percentage": [0.0],
    "last_transaction_year": [0.0],
    "last_transaction_month_sin": [0.0],
    "last_transaction_month_cos": [0.0],
    "last_transaction_day_sin": [0.0],
    "last_transaction_day_cos": [0.0],
    "expire_year": [0.0],
    "expire_month_sin": [0.0],
    "expire_month_cos": [0.0],
    "expire_day_sin": [0.0],
    "expire_day_cos": [0.0],
    "registration_year": [0.0],
    "registration_month_sin": [0.0],
    "registration_month_cos": [0.0],
    "registration_day_sin": [0.0],
    "registration_day_cos": [0.0],
        })

    def download_template_csv():
        csv = template_data.to_csv(index=False)
        return csv


    csv_template =download_template_csv()

    ste.sidebar.download_button(
        label="Download CSV Template",
        key="download_template_csv",
        data=csv_template,
        file_name="download_template.csv",
        mime="text/csv",
    )

    # st.sidebar.image('KKBOX_WW.png', use_column_width=True)




    def categorize_churn_risk(churn_probabilities):
        # Categorize users based on predicted churn percentages
        risk_categories = []
        for prob in churn_probabilities:
            if prob >= 90:
                risk_categories.append("High Risk")
            elif prob >= 50:
                risk_categories.append("Medium Risk")
            else:
                risk_categories.append("Low Risk")
        return risk_categories



    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # test with 10 rows-- to delete later
        ids= df.msno
        # X_test= df.drop(['msno', 'is_churn','bd','payment_method_id', 'city', 'registered_via'], axis=1)
        X_test= df.drop(['msno'], axis=1)

        if st.button("Predict"):

            X_columns = X_test.columns.to_list()
            X_transformed = loaded_preproc.fit_transform(X_test)
            X_transformed = pd.DataFrame(X_transformed,columns=X_columns)

            # make predictions
            predict = loaded_model.predict_proba(X_transformed)*100
            pred = predict[:,1].astype(float)
            new = pd.DataFrame({'ID': ids, 'Churn percentage': pred})

            random_indices = np.random.choice(len(new), 10, replace=False)
            random_rows = new.iloc[random_indices]

            st.subheader("Predictions:")
            formatted_df = random_rows.style.format({"Churn percentage": "{:.2f}".format})
            st.dataframe(formatted_df, use_container_width=True, hide_index=True)

            # st.dataframe(random_rows,use_container_width=True, hide_index=True)


            # Pie Chart
            st.subheader("Churn Statistics")

            churn_count = new['Churn percentage'].apply(lambda x: 'High risk Churn' if x >= 90 else ('Medium Risk Churn' if x >= 50 else 'Low Risk Churn') )
            churn_counts = churn_count.value_counts()

            fig = px.pie(new, values=churn_counts.values, names=churn_counts.index, color=churn_counts.index,
                         title="Churn Distribution",
             color_discrete_map={'High risk Churn':'#EA5F89',
                                 'Medium Risk Churn':'#F7B7A3',
                                 'Low Risk Churn':'#8bd3c7',
                                 })

            fig.update_traces(
                textposition='outside',
                textinfo='percent+label',
                pull=[0.1, 0.1],
                hole=0.3,
                outsidetextfont=dict(size=14),
                # marker=dict(colors=custom_colors)
                )
            fig.update_layout(
                    title_font=dict(size=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(font=dict(size=14)),
                    margin=dict(l=120, r=0, b=0, t=80),
                    autosize=True,
                    width = 800,
                    height = 600
                )

            st.plotly_chart(fig)


            churn_risks = categorize_churn_risk(new['Churn percentage'])

            # Create a DataFrame with user IDs and their corresponding churn risks
            risk_df = pd.DataFrame({'ID': ids, 'Churn Risk': churn_risks})

            high_risk_df = risk_df[risk_df['Churn Risk'] == 'High Risk']
            medium_risk_df = risk_df[risk_df['Churn Risk'] == 'Medium Risk']
            low_risk_df = risk_df[risk_df['Churn Risk'] == 'Low Risk']

            # Display risk categories in three separate columns


            st.subheader("Churn Risk Categories:")

            col1, col2, col3 = st.columns(3)

            with col1:
                with st.expander("HIGH RISK LIST", expanded=False):

                    st.subheader('High Risk 🔴')
                    new_df_c1 = pd.DataFrame(high_risk_df["ID"])
                    col_1 = new_df_c1.to_csv()
                    # st.header(3, f'Users with a churn percentage greater than 90% are considered to be high risk cases')
                    # st.write(f'Number of high risk cases: {len(new_df_c1)}')
                    st.markdown(f"""
                            - These are cases where the predicted probability of the target event (e.g., churn)
                            is relatively high, indicating a strong likelihood of the event occurring.
                            - Users with a churn percentage greater than or equal to 90%
                            are considered to be high risk cases
                            - High-risk cases often warrant immediate attention or
                            intervention because they are most likely to result in the event of interest.
                            - Number of high risk cases: {len(new_df_c1)}

                            """)
                    st.dataframe(new_df_c1.head(10),use_container_width=True, hide_index=True)


                    ste.download_button(
                        label="Download data as CSV",
                        data=col_1,
                        file_name='High_risk_df.csv',
                        mime='text/csv',
                    )

            with col2:
                 with st.expander("MEDIUM RISK LIST", expanded=False):
                    st.subheader('Medium Risk 🟠')
                    new_df_c2 = pd.DataFrame(medium_risk_df["ID"])
                    col_2 = new_df_c2.to_csv(index=False, encoding='utf-8')

                    st.markdown(f"""
                            - These are cases where the predicted probability of the target
                            event falls in an intermediate range.
                            - Users with a churn percentage greater than or equal to 50%
                            are considered to be medium risk cases
                            - Medium-risk cases require monitoring and some level of
                            intervention or retention efforts.
                            - Number of medium risk cases: {len(new_df_c2)}

                            """)

                    st.dataframe(new_df_c2.head(10),use_container_width=True, hide_index=True )

                    ste.download_button(
                        label="Download data as CSV",
                        data=col_2,
                        file_name='Medium_risk_df.csv',
                        mime='text/csv',
                    )

            with col3:
                 with st.expander("LOW RISK LIST", expanded=False):
                    st.subheader('Low Risk 🟢')
                    new_df_c3 = pd.DataFrame(low_risk_df["ID"])
                    col_3 = new_df_c3.to_csv(index=False, encoding='utf-8')
                    st.markdown(f"""
                            - These are cases where the predicted probability of the target event is relatively
                            low, suggesting a low likelihood of the event occurring.
                            - Users with a churn percentage less than 50% are considered to be low risk cases
                            - Low-risk cases typically require less immediate attention and may not be the top
                            priority for retention efforts.
                            - Number of low risk cases: {len(new_df_c3)}

                            """)
                    st.dataframe(new_df_c3.head(10),use_container_width=True, hide_index=True )

                    ste.download_button(
                        label="Download data as CSV",
                        data=col_3,
                        file_name='Low_risk_df.csv',
                        mime='text/csv',
                    )


        # Hide streamlit style

        hide_st_style = """
        <style>
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        # header {visibility:hidden;}
        </style>"""

        st.markdown(hide_st_style, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
