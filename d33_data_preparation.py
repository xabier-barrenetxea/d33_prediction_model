# Databricks notebook source
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
-- Load dates for aggregating data on different dates
WITH dates AS (
    SELECT date_dt
    FROM deep_purple.etl_glossary.dim_date
    WHERE date_dt >= '2023-01-01'
      AND date_dt < CURRENT_DATE
),

-- Cross join student info with dates
users_w_dates AS (
    SELECT s.id student_id
         , d.date_dt
         , s.first_payment_ts::date first_payment_date
         , DATEDIFF('DAYS',first_payment_date, d.date_dt) student_age
         , s.first_subscription_package_ts::date first_sub_payment_date
         , DATEDIFF('DAYS',first_sub_payment_date, d.date_dt)  days_since_first_sub_payment
         , s.country_code 
         , s.focus_countries 
         , s.preply_region 
         , s.projected_gross_margin_360d 
         , s.language_version 
    FROM dates d
    CROSS JOIN deep_purple.cds.dim_student s
    WHERE d.date_dt >= s.first_payment_ts::date
    	AND first_subscription_package_ts IS NOT NULL
    	AND d.date_dt <= DATEADD('DAYS',33,s.first_subscription_package_ts::date) 
    	AND s.is_staff = FALSE 
        AND s.first_subscription_package_ts::date <= d.date_dt
),

-- Data of number of active subs on a given date
total_active_sub AS (
	SELECT user_id
		, subscription_log_date 
		, COUNT(DISTINCT tutoring_id) total_active_subs
	FROM deep_purple.cds.agg_daily_subscription_logs 
	WHERE subscription_log_date <= DATEADD('DAYS',33,first_subscription_event_ts::date) 
		AND subscription_active_day = TRUE
	GROUP BY 1,2
),

-- Filter out students with only one active sub
only_one_active_sub AS (
	SELECT user_id
		, MAX(total_active_subs) max_total_active_subs
	FROM total_active_sub
	GROUP BY 1
	HAVING max_total_active_subs = 1
),

-- Apply the filter with the users and date tmp table
users_w_dates_one_sub AS (
	SELECT uw.*
	FROM users_w_dates uw
	JOIN only_one_active_sub oas
		ON uw.student_id = oas.user_id

),

-- Get session info aggregations
session_info AS (
	SELECT ud.student_id
		, ud.date_dt
		, SUM(ds.pages_viewed) sum_pages_viewed_all
		, AVG(ds.pages_viewed) avg_pages_viewed_all
		, COUNT(DISTINCT ds.start_ts::DATE) days_with_session
		, SUM(ds.duration_sec) sum_session_duration_all
		, AVG(ds.duration_sec) avg_session_duration_all
		, SUM(CASE WHEN ds.app_version IS NOT NULL THEN ds.pages_viewed END) sum_page_viewed_app
		, AVG(CASE WHEN ds.app_version IS NOT NULL THEN ds.pages_viewed END) avg_page_viewed_app
		, SUM(CASE WHEN ds.app_version IS NOT NULL THEN ds.duration_sec END) sum_session_duration_app
		, AVG(CASE WHEN ds.app_version IS NOT NULL THEN ds.duration_sec END) avg_session_duration_app
		, SUM(CASE WHEN ds.app_version IS NULL THEN ds.pages_viewed END) sum_page_viewed_non_app
		, AVG(CASE WHEN ds.app_version IS NULL THEN ds.pages_viewed END) avg_page_viewed_non_app
		, SUM(CASE WHEN ds.app_version IS NULL THEN ds.duration_sec END) sum_session_duration_non_app
		, AVG(CASE WHEN ds.app_version IS NULL THEN ds.duration_sec END) avg_session_duration_non_app
		, DATEDIFF('DAY',MAX(ds.start_ts), date_dt) days_since_last_session
	FROM users_w_dates_one_sub  ud
	LEFT JOIN deep_purple.cds.dim_session ds 
		ON ud.student_id = ds.user_id
		AND ds.start_ts::date <= ud.date_dt
		AND ds.start_ts::date >=  ud.first_sub_payment_date
	GROUP BY 1,2
),

-- Message information
message_info AS (
	SELECT ud.student_id
		, ud.date_dt
		, COUNT(CASE WHEN tm.user_id = tmf.author_id THEN tmf.time_posted END) student_messages_sent
		, COUNT(CASE WHEN tm.tutor_id = tmf.author_id THEN tmf.time_posted END) tutor_messages_sent
	FROM users_w_dates_one_sub ud 
	LEFT JOIN deep_purple.dds.tutors_messagethread tm
		ON tm.user_id = ud.student_id
	LEFT JOIN deep_purple.dds.tutors_message tmf 
		ON tm.id = tmf.thread_id 
		AND tmf.time_posted::date <= ud.date_dt
		AND tmf.time_posted::date >=  ud.first_sub_payment_date
		AND tmf.author_id IS NOT NULL
	GROUP BY 1,2
),

-- Lesson information
lesson_info AS (
	SELECT ud.student_id
		, ud.date_dt
		, MAX(dl.lesson_ts::date) last_confirmed_lesson_date
		, DATEDIFF('DAYS',last_confirmed_lesson_date,ud.date_dt) n_days_since_last_confirmed_lesson
		, COUNT(DISTINCT dl.id) num_confirmed_lessons_during_sub
		, AVG(COALESCE(dl.rating_amount,0)) avg_lesson_rating
	FROM users_w_dates_one_sub ud
	LEFT JOIN deep_purple.cds.dim_lesson dl 
		ON ud.student_id = dl.student_id 
		AND dl.lesson_ts::date <= ud.date_dt
		AND dl.lesson_ts::date >=  ud.first_sub_payment_date
		AND dl.lesson_status IN ('COMPLETED','AUTOCOMPLETED')
	GROUP BY 1,2
),

-- Sub Payment information
payment_info AS (
	SELECT ud.student_id
		, ud.date_dt
	    , fp.hours
	    , fp.subject
	    , fp.gmv_proceeds_usd
	FROM users_w_dates_one_sub ud
	LEFT JOIN deep_purple.cds.fact_payment fp
		ON ud.student_id = fp.user_id 
		AND fp.user_subscription_package_order = 1
		AND fp.tutoring_subscription_package_order = 1
),

-- Tutor search information
search_info AS (
	SELECT ud.student_id
		, ud.date_dt
		, SUM(CASE WHEN fev.event_name = 'search_completed' THEN 1 ELSE 0 END) AS num_search_page_views
		, SUM(CASE WHEN fev.event_name = 'search_card_viewed' THEN 1 ELSE 0 END) AS num_impressions	
	FROM users_w_dates_one_sub ud
	LEFT JOIN deep_purple.cds.fact_event_v2 fev
		ON ud.student_id = fev.user_id 
		AND fev.event_ts::date <= ud.date_dt
		AND fev.event_ts::date >=  ud.first_sub_payment_date
		AND fev.event_name IN ('search_completed','search_card_viewed')
	GROUP BY 1,2
),

-- Sub status information
sub_status_info_day_33 AS (
	SELECT ud.student_id
		, ud.date_dt
		, adsl2.subscription_active_day
		, adsl1.subscription_active_day subscription_active_day_d33
	FROM users_w_dates_one_sub ud
	LEFT JOIN deep_purple.cds.agg_daily_subscription_logs adsl1
		ON ud.student_id = adsl1.user_id 
		AND DATEADD('DAYS',33,ud.first_sub_payment_date) = adsl1.subscription_log_date 
	LEFT JOIN deep_purple.cds.agg_daily_subscription_logs adsl2
		ON ud.student_id = adsl2.user_id 
		AND ud.date_dt = adsl2.subscription_log_date 
),

-- All features join
all_features AS (
	SELECT ud.student_id
        , ud.date_dt
        , ud.first_payment_date
        , ud.student_age
        , ud.first_sub_payment_date
        , ud.days_since_first_sub_payment
        , ud.country_code 
        , ud.focus_countries 
        , ud.preply_region 
        , ud.projected_gross_margin_360d 
        , ud.language_version 
		, ssid.subscription_active_day
		, CASE WHEN ssid.subscription_active_day_d33 = TRUE THEN FALSE ELSE TRUE END AS churn_day_33
		, li.last_confirmed_lesson_date
		, li.n_days_since_last_confirmed_lesson
		, li.num_confirmed_lessons_during_sub
		, li.avg_lesson_rating
		, si.sum_pages_viewed_all
		, si.avg_pages_viewed_all
		, si.days_with_session
		, si.sum_session_duration_all
		, si.avg_session_duration_all
		, si.sum_page_viewed_app
		, si.avg_page_viewed_app
		, si.sum_session_duration_app
		, si.avg_session_duration_app
		, si.sum_page_viewed_non_app
		, si.avg_page_viewed_non_app
		, si.sum_session_duration_non_app
		, si.avg_session_duration_non_app
		, si.days_since_last_session
		, mi.student_messages_sent
		, mi.tutor_messages_sent
	    , pi.hours
	    , pi.subject
	    , pi.gmv_proceeds_usd
	    , sri.num_search_page_views
		, sri.num_impressions	
	FROM users_w_dates_one_sub ud
	JOIN sub_status_info_day_33 ssid
		ON ud.student_id = ssid.student_id
		AND ud.date_dt = ssid.date_dt
	JOIN lesson_info li 
		ON ud.student_id = li.student_id
		AND ud.date_dt = li.date_dt	
	JOIN session_info si
		ON ssid.student_id = si.student_id
		AND ssid.date_dt = si.date_dt
	JOIN message_info mi
		ON ssid.student_id = mi.student_id
		AND ssid.date_dt = mi.date_dt
	JOIN payment_info pi
		ON ssid.student_id = pi.student_id
		AND ssid.date_dt = pi.date_dt
	JOIN search_info sri
		ON ssid.student_id = sri.student_id
		AND ssid.date_dt = sri.date_dt
)

-- Generate data
SELECT *
FROM all_features 
QUALIFY ROW_NUMBER() OVER (PARTITION BY student_id, date_dt ORDER BY SUBSCRIPTION_ACTIVE_DAY ) = 1
"""

# COMMAND ----------

df = read_table(query = query)

# COMMAND ----------

df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("churn_model_d33_training_data")
