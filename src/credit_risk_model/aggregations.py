import duckdb

# Aggregate bureau_balance
def get_agg_bureau_balance(con):
    con.execute("""
     CREATE OR REPLACE TABLE agg_bureau_balance AS
        SELECT 
            SK_ID_BUREAU,
            COUNT(*) AS no_of_months,
            SUM(STATUS='1') AS bureau_n_late_months,
            SUM(STATUS IN ('2','3','4','5')) AS bureau_n_very_late_months,
            SUM(STATUS='0') AS bureau_n_on_time,
            SUM(STATUS='C') AS bureau_n_closed,
            SUM(STATUS='X') AS bureau_n_no_info
        FROM bureau_balance
        GROUP BY SK_ID_BUREAU
    """)
    print("created agg_bureau_balance table")


# Join bureau with aggregated balance
def get_join_bureau_df(con):    
    con.execute("""
        CREATE OR REPLACE TABLE join_bureau_df AS
        SELECT 
            b.SK_ID_CURR,
            COUNT(*) AS n_loans,
            SUM(CREDIT_ACTIVE='Active') AS n_active_loans,
            SUM(AMT_CREDIT_SUM_DEBT) AS total_debt,
            AVG(AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM) AS avg_debt_ratio,
            SUM(bb.bureau_n_late_months) AS total_late_months,
            SUM(bb.bureau_n_late_months) * 1.0 / NULLIF(SUM(bb.no_of_months),0) AS pct_months_late,
            MAX(bb.bureau_n_late_months) AS max_late_single_loan
        FROM bureau AS b
        LEFT JOIN agg_bureau_balance AS bb 
            ON b.SK_ID_BUREAU = bb.SK_ID_BUREAU
        GROUP BY SK_ID_CURR
     """)
    print("created join_bureau_df table")

def get_credit_card_balance(con):
    con.execute("""
    CREATE OR REPLACE TABLE agg_credit_card_balance AS
    SELECT 
        SK_ID_CURR,
        AVG(AMT_BALANCE) AS avg_balance,
        MAX(AMT_BALANCE) AS max_balance,
        AVG(AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL) AS avg_utilization
    FROM credit_card_balance
    GROUP BY SK_ID_CURR
    """)
    print("created agg_credit_card_balance table")

def get_previous_application(con):
    con.execute("""
    CREATE OR REPLACE TABLE agg_previous_application AS
    SELECT 
        SK_ID_CURR,
        COUNT(*) AS n_prev_apps,
        SUM(NAME_CONTRACT_STATUS = 'Approved') as n_approved,
        SUM(NAME_CONTRACT_STATUS = 'Refused') as n_refused,
        SUM(NAME_CONTRACT_STATUS = 'Canceled') as n_cancelled,
        AVG(AMT_APPLICATION) AS avg_amt_applied,
        AVG(AMT_CREDIT) AS avg_amt_credit,
        SUM(NAME_CONTRACT_STATUS = 'Approved') * 1.0 / COUNT(*) AS approval_rate
    FROM previous_application
    GROUP BY SK_ID_CURR
    """)
    print("created agg_previous_application table")

def get_installments_payments(con):
    con.execute("""
        CREATE OR REPLACE TABLE agg_installments_payments AS
        SELECT
            SK_ID_CURR,
            AVG(AMT_PAYMENT/ AMT_INSTALMENT) as avg_payment_ratio,
            SUM(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END) AS installments_n_late_payments
        FROM installments_payments
        GROUP BY SK_ID_CURR
    """)
    print("created agg_installments_payments table")


def get_pos_cash_balance(con):
        con.execute("""
        CREATE OR REPLACE TABLE agg_pos_cash_balance AS
        SELECT
            SK_ID_CURR,
            SUM(NAME_CONTRACT_STATUS = 'Active') as n_active_contracts,
            AVG(MONTHS_BALANCE) FILTER (WHERE NAME_CONTRACT_STATUS = 'Active') as avg_months_active,
            SUM(NAME_CONTRACT_STATUS='Completed') AS n_completed_contracts
        FROM POS_CASH_balance
        GROUP BY SK_ID_CURR
    """)
        print("created agg_pos_cash_balance table")

def get_agg_application_train(con):
    con.execute("""
    CREATE OR REPLACE TABLE agg_application_train AS               
    SELECT 
        SK_ID_CURR,
        TARGET AS target,
        (AMT_CREDIT / AMT_INCOME_TOTAL) AS credit_income_ratio,
        AMT_INCOME_TOTAL AS total_income,
        AMT_CREDIT AS total_credit_requested,
        AMT_ANNUITY AS monthly_loan_payment,
        AMT_GOODS_PRICE AS value_of_goods_financed,
        (DAYS_BIRTH / -365) AS age_years,
        (DAYS_EMPLOYED / -365) AS employment_years,
        NAME_CONTRACT_TYPE,
        CODE_GENDER,
        OCCUPATION_TYPE
    FROM application_train 
   """)
    print("created agg_application_train table")

def get_agg_main(con):
        con.execute("""
        CREATE OR REPLACE TABLE agg_main AS
        SELECT 
            a.*,
            b.* EXCLUDE (SK_ID_CURR),
            c.* EXCLUDE (SK_ID_CURR),
            d.* EXCLUDE (SK_ID_CURR),
            e.* EXCLUDE (SK_ID_CURR),
            f.* EXCLUDE (SK_ID_CURR)
        FROM agg_application_train a
        LEFT JOIN join_bureau_df b USING(SK_ID_CURR)
        LEFT JOIN agg_previous_application c USING(SK_ID_CURR)
        LEFT JOIN agg_pos_cash_balance d USING(SK_ID_CURR)
        LEFT JOIN agg_installments_payments e USING(SK_ID_CURR)
        LEFT JOIN agg_credit_card_balance f USING(SK_ID_CURR);
           """)
        print("created agg_main table")


# Execute all aggregation functions in order
def run_all_aggregations(con):
    get_agg_bureau_balance(con)
    get_join_bureau_df(con)
    get_credit_card_balance(con)
    get_previous_application(con)
    get_installments_payments(con)
    get_pos_cash_balance(con)
    get_agg_application_train(con)
    get_agg_main(con)

    df = con.execute("SELECT * FROM agg_main").df()
    print("Aggregation completed and agg_main table created.")
    return df
