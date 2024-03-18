# Databricks notebook source
import pandas as pd

# COMMAND ----------

sfUser = dbutils.secrets.get(scope="snowflake", key="username")
sfPassword = dbutils.secrets.get(scope="snowflake", key="password")
sfUrl = dbutils.secrets.get(scope="snowflake", key="host")

#add secrets
options = {
  "sfUrl": sfUrl,
  "sfUser": sfUser,
  "sfPassword": sfPassword,
  "sfDatabase": "DEEP_PURPLE",
  "sfSchema": "EDS",
  "sfWarehouse": "DATABRICKS_WH"
}

# read table function (table name in -> Spark Dataframe out)
# .option("dbtable", table_name) \
def read_table(table_name = None, condition = None, query = None):
    if condition is None and query is None and table_name is not None:
        df = spark.read \
        .format("snowflake") \
        .options(**options) \
        .option("query", f"select * from {table_name}") \
        .load()
        return df
    elif query is None and table_name is not None:
        df = spark.read \
        .format("snowflake") \
        .options(**options) \
        .option("query", f"select * from {table_name} WHERE {condition}") \
        .load()
        return df
    else:
        df = spark.read \
        .format("snowflake") \
        .options(**options) \
        .option("query", query) \
        .load()
        return df 

# COMMAND ----------

query = """
WITH dates AS (
    SELECT date_dt
    FROM deep_purple.etl_glossary.dim_date
    WHERE date_dt >= '2023-01-01'
      AND date_dt < CURRENT_DATE
),

users_w_dates AS (
    SELECT s.id student_id
         , d.date_dt
         , s.first_payment_ts::date first_payment_date
         , DATEDIFF('DAYS',first_payment_date, d.date_dt) student_age
         , s.first_subscription_package_ts::date first_sub_payment_date
         , DATEDIFF('DAYS',first_sub_payment_date, d.date_dt)  days_since_first_sub_payment
    FROM dates d
    CROSS JOIN deep_purple.cds.dim_student s
    WHERE d.date_dt >= s.first_payment_ts::date
    	AND first_subscription_package_ts IS NOT NULL
    	AND d.date_dt <= DATEADD('DAYS',33,first_sub_payment_date) 
    	AND s.is_staff = FALSE 
    	AND s.is_package_student 
        AND s.first_subscription_package_ts::date <= d.date_dt
),

total_active_sub AS (
	SELECT user_id
		, subscription_log_date 
		, COUNT(DISTINCT tutoring_id) total_active_subs
	FROM deep_purple.cds.agg_daily_subscription_logs 
	WHERE subscription_log_date <= DATEADD('DAYS',33,first_subscription_event_ts::date) 
		AND subscription_active_day = TRUE
	GROUP BY 1,2
),

only_one_active_sub AS (
	SELECT user_id
		, MAX(total_active_subs) max_total_active_subs
	FROM total_active_sub
	GROUP BY 1
	HAVING max_total_active_subs = 1
),

users_w_dates_one_sub AS (
	SELECT uw.*
	FROM users_w_dates uw
	JOIN only_one_active_sub oas
		ON uw.student_id = oas.user_id

),

lesson_info AS (
	SELECT ud.student_id
		, ud.date_dt
		, ud.first_payment_date
		, ud.student_age
		, ud.first_sub_payment_date
		, ud.days_since_first_sub_payment
		, MAX(dl.lesson_ts::date) last_confirmed_lesson_date
		, DATEDIFF('DAYS',last_confirmed_lesson_date,ud.date_dt) n_days_since_last_confirmed_lesson
		, COUNT(DISTINCT dl.id) num_confirmed_lessons_during_sub
	FROM users_w_dates_one_sub ud
	LEFT JOIN deep_purple.cds.dim_lesson dl 
		ON ud.student_id = dl.student_id 
		AND dl.lesson_ts::date <= date_dt
		AND dl.lesson_ts::date >=  ud.first_sub_payment_date
		AND dl.lesson_status IN ('COMPLETED','AUTOCOMPLETED')
	GROUP BY 1,2,3,4,5,6
),

sub_status_info_day_33 AS (
	SELECT li.*
		, adsl2.subscription_active_day
		, adsl1.subscription_active_day subscription_active_day_d33
	FROM lesson_info li
	LEFT JOIN deep_purple.cds.agg_daily_subscription_logs adsl1
		ON li.student_id = adsl1.user_id 
		AND DATEADD('DAYS',33,li.first_sub_payment_date) = adsl1.subscription_log_date 
	LEFT JOIN deep_purple.cds.agg_daily_subscription_logs adsl2
		ON li.student_id = adsl1.user_id 
		AND li.date_dt = adsl2.subscription_log_date 
)

SELECT lsdf.*
    , ssid.first_sub_payment_date
    , ssid.days_since_first_sub_payment
    , ssid.last_confirmed_lesson_date
    , ssid.n_days_since_last_confirmed_lesson
    , ssid.num_confirmed_lessons_during_sub
    , ssid.subscription_active_day
    , ssid.subscription_active_day_d33
FROM sub_status_info_day_33 ssid
JOIN deep_purple.eds.ltv_subs_daily_features lsdf
    ON ssid.student_id = lsdf.student_id
    AND ssid.student_age = lsdf.student_age
"""

# COMMAND ----------

df = read_table(query = query)

# COMMAND ----------

display(df)

# COMMAND ----------


