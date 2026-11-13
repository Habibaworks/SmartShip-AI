
from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_FILE = BASE_DIR / "smartship_model.pkl"

st.set_page_config(
    page_title="SmartShip AI | Delivery Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PROFESSIONAL DESIGN
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

:root {
    --ink: #152946;
    --muted: #65758d;
    --accent: #1366d6;
    --border: #e6edf5;
    --bg: #f5f8fc;
}

html, body, [class*="css"], [data-testid="stApp"] {
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 1.8rem;
    max-width: 1350px;
}

[data-testid="stSidebar"] {
    background: #102541;
}

[data-testid="stSidebar"] * {
    color: #edf5ff !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: .45rem .65rem;
    border-radius: 10px;
}

h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    color: var(--ink);
    letter-spacing: -.025em;
}

h1 {
    font-size: 2.25rem !important;
}

[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 1.2rem 1.3rem;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.1rem 1.15rem;
    min-height: 115px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
}

[data-testid="stMetricValue"] {
    color: var(--ink);
    font-family: 'Outfit', sans-serif;
}

[data-testid="stFormSubmitButton"] > button {
    background: #1769dc !important;
    color: white !important;
    border: 0 !important;
    border-radius: 11px !important;
    font-weight: 700 !important;
}

[data-testid="stAlert"] {
    border-radius: 13px;
}

.hero {
    background: linear-gradient(
        110deg, #102b4c 5%, #155aaa 72%, #2c85dd
    );
    border-radius: 20px;
    padding: 32px 34px;
    color: white;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.hero:after {
    content: '◉';
    position: absolute;
    right: 60px;
    top: -77px;
    color: rgba(255,255,255,.07);
    font-size: 290px;
}

.hero .eyebrow {
    color: #b7d6ff;
    letter-spacing: .17em;
    text-transform: uppercase;
    font-size: .72rem;
    font-weight: 700;
}

.hero h1 {
    color: white !important;
    font-size: 2.35rem !important;
    margin: .35rem 0;
}

.hero p {
    color: #d8e9ff;
    max-width: 670px;
    line-height: 1.65;
}

.section-caption {
    font-size: .78rem;
    letter-spacing: .13em;
    text-transform: uppercase;
    font-weight: 800;
    color: #3976c8;
    margin: 10px 0 12px;
}

.note {
    font-size: .88rem;
    color: #62748e;
    line-height: 1.55;
}

.result-card {
    background: white;
    border: 1px solid #e1eaf5;
    border-radius: 16px;
    padding: 22px 24px;
    margin: 12px 0;
}

.result-card .state {
    font-family: 'Outfit', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #153456;
}

.result-card .copy {
    color: #5f7188;
    margin: 7px 0 0;
}

.pill {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 800;
}

.pill.high {
    color: #9f2e2e;
    background: #ffe8e8;
}

.pill.low {
    color: #156046;
    background: #ddf7eb;
}

.brand {
    background: rgba(255,255,255,.09);
    padding: 18px 16px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,.14);
    margin: 9px 0 24px;
}

.brand-name {
    font: 800 1.48rem 'Outfit', sans-serif;
    color: white;
}

.brand-caption {
    font-size: .8rem;
    color: #b7cceb;
    margin-top: 3px;
}

.footer {
    font-size: .78rem;
    color: #8390a3;
    margin: 26px 0 8px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD MODEL AND DATA
# ==========================================

@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


@st.cache_data
def load_tables(data_dir, file_times):
    customers = pd.read_csv(
        data_dir / "customer_state_centroids.csv",
        index_col=0
    )

    seller_states = pd.read_csv(
        data_dir / "seller_state_centroids.csv",
        index_col=0
    )

    cats = (
        pd.read_csv(
            data_dir / "product_category_name_translation.csv"
        )["product_category_name_english"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    
    seller_stats = pd.read_csv(
        data_dir / "seller_stats.csv"
    )

    zip_centroids = pd.read_csv(
        data_dir / "zip_centroids.csv"
    ).set_index("geolocation_zip_code_prefix")

    zip_states = pd.read_csv(
        data_dir / "zip_states.csv"
    ).set_index("geolocation_zip_code_prefix")

    return (
        customers,
        seller_states,
        cats,
        seller_stats,
        zip_centroids,
        zip_states
    )


required_files = [
    MODEL_FILE,
    DATA_DIR / "customer_state_centroids.csv",
    DATA_DIR / "seller_state_centroids.csv",
    DATA_DIR / "product_category_name_translation.csv",
    DATA_DIR / "seller_stats.csv",
    DATA_DIR / "zip_centroids.csv",
    DATA_DIR / "zip_states.csv",
]
missing = [
    p for p in required_files if not p.is_file()
]

if missing:
    st.error("Some required project files are missing.")
    st.code("\n".join(str(p) for p in missing))
    st.stop()

try:
    bundle = load_model(
        str(MODEL_FILE),
        MODEL_FILE.stat().st_mtime_ns
    )

    cust, sell, cats, sellers, zip_centroids, zip_states = load_tables(
        DATA_DIR,
        tuple(
            (p.name, p.stat().st_mtime_ns)
            for p in required_files[1:]
        )
    )

    model = bundle["model"]
    threshold = float(bundle["threshold"])

    features = (
        list(bundle["num_features"])
        + list(bundle["cat_features"])
    )

    if not (0 < threshold < 1):
        raise ValueError("Invalid saved threshold")

except Exception as exc:
    st.error("Could not load model or supporting data.")
    st.exception(exc)
    st.stop()
# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def haversine(lat1, lon1, lat2, lon2):
    vals = np.radians(
        np.asarray(
            [lat1, lon1, lat2, lon2],
            dtype=float
        )
    )

    lat1, lon1, lat2, lon2 = vals

    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin((lon2 - lon1) / 2) ** 2
    )

    return float(
        2 * 6371
        * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )


def heading(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">
                {html.escape(eyebrow)}
            </div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 5. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-name">📦 SmartShip AI</div>
            <div class="brand-caption">
                Delivery Intelligence Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "WORKSPACE",
        [
            "📦  Order risk predictor",
            "📊  Seller performance"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("MODEL STATUS")
    st.success("Model loaded")

    st.caption(
        f"Decision threshold: {threshold:.4f}"
    )

    st.caption(
        "Olist Brazilian E-Commerce Dataset"
    )

    st.divider()

    st.caption(
        "Portfolio project | Streamlit & scikit-learn"
    )


# ==========================================
# 6. ORDER RISK PREDICTION PAGE
# ==========================================

if page.startswith("📦"):

    heading(
        "Predictive operations",
        "Know the risk before you ship.",
        "Assess delivery-delay risk using product, "
        "shipping and order information."
    )

    st.markdown(
        '<div class="section-caption">01 / Order details</div>',
        unsafe_allow_html=True
    )

    # State selectors sit outside the form so changing a state immediately
    # refreshes its available ZIP prefixes, without requiring a submission.
    zip_lookup = zip_states[["state"]].join(
        zip_centroids[["lat", "lng"]], how="inner"
    )
    zip_lookup = zip_lookup.dropna(subset=["state", "lat", "lng"])
    zip_lookup = zip_lookup[~zip_lookup.index.duplicated(keep="first")]
    zip_lookup.index = zip_lookup.index.astype(int)
    zip_lookup["state"] = zip_lookup["state"].astype(str)

    seller_states = sorted(
        set(sell.index.dropna().astype(str)) & set(zip_lookup["state"])
    )
    customer_states = sorted(
        set(cust.index.dropna().astype(str)) & set(zip_lookup["state"])
    )
    if not seller_states or not customer_states:
        st.error("No usable ZIP/state mappings are available for prediction.")
        st.stop()

    st.markdown("#### Seller and customer locations")
    state_col1, state_col2 = st.columns(2, gap="large")
    with state_col1:
        seller_state = st.selectbox(
            "Seller state",
            seller_states,
            index=seller_states.index("SP") if "SP" in seller_states else 0,
        )
    with state_col2:
        customer_state = st.selectbox(
            "Customer state",
            customer_states,
            index=customer_states.index("SP") if "SP" in customer_states else 0,
        )

    seller_zips = sorted(zip_lookup.index[zip_lookup["state"] == seller_state].tolist())
    customer_zips = sorted(zip_lookup.index[zip_lookup["state"] == customer_state].tolist())

    with st.form("order_form"):

        st.markdown("#### Product & shipping")

        c1, c2, c3 = st.columns(
            3,
            gap="large"
        )

        with c1:

            default_cat = (
                cats.index("bed_bath_table")
                if "bed_bath_table" in cats
                else 0
            )

            category = st.selectbox(
                "Product category",
                cats,
                index=default_cat
            )

            price = st.number_input(
                "Total product price (BRL)",
                min_value=1.0,
                max_value=20000.0,
                value=100.0,
                step=10.0
            )

            freight = st.number_input(
                "Freight cost (BRL)",
                min_value=0.0,
                max_value=2000.0,
                value=20.0,
                step=5.0
            )

        with c2:

            n_items = st.number_input(
                "Items in order",
                min_value=1,
                max_value=20,
                value=1
            )

            weight = st.number_input(
                "Representative product weight (g)",
                min_value=1.0,
                max_value=40000.0,
                value=1000.0
            )

            volume = st.number_input(
                "Representative product volume (cm³)",
                min_value=100.0,
                max_value=300000.0,
                value=5000.0
            )

        with c3:

            # Only ZIP prefixes present in the Olist geolocation tables
            # and compatible with the selected state can be chosen.
            seller_zip = st.selectbox(
                "Seller ZIP prefix",
                seller_zips,
                index=seller_zips.index(1310) if 1310 in seller_zips else 0,
                format_func=lambda z: f"{z:05d}",
                help="First five digits of a Brazilian ZIP code, from the dataset.",
            )

            customer_zip = st.selectbox(
                "Customer ZIP prefix",
                customer_zips,
                index=customer_zips.index(1310) if 1310 in customer_zips else 0,
                format_func=lambda z: f"{z:05d}",
                help="First five digits of a Brazilian ZIP code, from the dataset.",
            )

            est_days = st.slider(
                "Estimated delivery days",
                min_value=2,
                max_value=155,
                value=23
            )

        st.divider()

        st.markdown("#### Purchase timing")

        c4, c5 = st.columns(2)

        with c4:
            purchase_date = st.date_input(
                "Purchase date"
            )

        with c5:
            purchase_time = st.time_input(
                "Purchase time"
            )

        submitted = st.form_submit_button(
            "Analyze delivery risk",
            type="primary",
            use_container_width=True
        )

    st.caption(
        "The model was trained on historical delivered "
        "orders. For multi-item orders, product attributes "
        "represent a single product. The training data covers 2016–2018, "
        "so predictions for current orders are illustrative and need fresh-data validation."
    )

    # ======================================
    # PREDICTION RESULTS
    # ======================================

    if submitted:

        try:

            if seller_zip not in zip_centroids.index:
                st.error("Seller ZIP Code not found in dataset.")
                st.stop()

            if customer_zip not in zip_centroids.index:
                st.error("Customer ZIP Code not found in dataset.")
                st.stop()
            
            if (
                seller_zip not in zip_states.index
                or customer_zip not in zip_states.index
            ):
                st.error("ZIP state information is unavailable.")
                st.stop()

            seller_zip_state = str(zip_states.loc[seller_zip, "state"])
            customer_zip_state = str(zip_states.loc[customer_zip, "state"])

            if seller_zip_state != seller_state:
                st.error(
                    "Seller ZIP Code does not match the selected seller state."
                )
                st.stop()

            if customer_zip_state != customer_state:
                st.error(
                    "Customer ZIP Code does not match the selected customer state."
                )
                st.stop()
    
            distance = haversine(
                zip_centroids.loc[seller_zip, "lat"],
                zip_centroids.loc[seller_zip, "lng"],
                zip_centroids.loc[customer_zip, "lat"],
                zip_centroids.loc[customer_zip, "lng"]
            )


            row = pd.DataFrame([{
                "price": price,
                "freight_value": freight,
                "n_items": n_items,
                "product_weight_g": weight,
                "distance_km": distance,
                "estimated_days": est_days,
                "freight_ratio": freight / (price + 1),
                "volume_cm3": volume,
                "purchase_month": purchase_date.month,
                "purchase_dow": purchase_date.weekday(),
                "purchase_hour": purchase_time.hour,
                "same_state": int(
                    seller_state == customer_state
                ),
                "customer_state": customer_state,
                "seller_state": seller_state,
                "product_category_name_english": category
            }])[features]

            score = float(
                model.predict_proba(row)[0, 1]
            )

            if not np.isfinite(score):
                raise ValueError("Invalid model score")

            high = score >= threshold

            st.divider()

            st.markdown(
                '<div class="section-caption">'
                '02 / Risk assessment</div>',
                unsafe_allow_html=True
            )

            a, b, c = st.columns(3)

            a.metric(
                "MODEL RISK SCORE",
                f"{score:.3f}"
            )

            b.metric(
                "ESTIMATED DISTANCE",
                f"{distance:,.0f} km"
            )

            c.metric(
                "RISK CLASSIFICATION",
                "High risk" if high else "Low risk"
            )

            st.progress(
                float(np.clip(score, 0, 1)),
                text="Model score (not a calibrated probability)"
            )

            if high:

                st.markdown(
                    """
                    <div class="result-card">
                        <span class="pill high">
                            ATTENTION REQUIRED
                        </span>
                        <div class="state">
                            Elevated late-delivery risk
                        </div>
                        <p class="copy">
                            Consider early customer communication
                            and prioritized carrier monitoring.
                            Avoid automatic paid upgrades without
                            reviewing operational costs.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="result-card">
                        <span class="pill low">
                            STANDARD MONITORING
                        </span>
                        <div class="state">
                            Lower relative delivery risk
                        </div>
                        <p class="copy">
                            Follow the normal fulfillment process
                            while keeping routine tracking in place.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.caption(
                f"Decision threshold: {threshold:.4f}. "
                "Selected using validation data for a recall-focused decision. "
                "Scores are not calibrated probabilities."
            )

            st.info(
                "Shipping distance is estimated using ZIP-code prefix "
                "coordinates and the Haversine formula. "
                "It represents straight-line distance, not actual road distance."
            )

        except Exception as exc:

            st.error(
                "Prediction could not be completed."
            )

            st.exception(exc)


# ==========================================
# 7. SELLER PERFORMANCE PAGE
# ==========================================

else:

    heading(
        "Seller intelligence",
        "Find patterns behind delivery delays.",
        "Compare historical seller-associated late rates, "
        "explore regions and export seller reports."
    )

    st.markdown(
        '<div class="section-caption">'
        '01 / Explore seller history</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        min_orders = st.slider(
            "Minimum orders per seller",
            min_value=1,
            max_value=200,
            value=30
        )

    with c2:
        options = (
            ["All"]
            + sorted(
                sellers["state"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        region = st.selectbox(
            "Seller state",
            options
        )

    view = sellers.loc[
        sellers["orders"] >= min_orders
    ].copy()

    if region != "All":
        view = view.loc[
            view["state"] == region
        ].copy()

    view = view.sort_values(
        ["late_rate", "orders"],
        ascending=[False, False]
    )

    all_orders = sellers["orders"].sum()

    overall = (
        float(
            (
                sellers["late_rate"] * sellers["orders"]
            ).sum() / all_orders
        )
        if all_orders > 0
        else np.nan
    )

    st.divider()

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "SELLERS MATCHING FILTERS",
        f"{len(view):,}"
    )

    m2.metric(
        "OVERALL SELLER-ORDER LATE RATE",
        f"{overall:.1%}"
        if np.isfinite(overall)
        else "N/A"
    )

    m3.metric(
        "HIGHEST RATE IN FILTER",
        f"{view['late_rate'].max():.1%}"
        if not view.empty
        else "N/A"
    )

    st.markdown(
        '<div class="section-caption">'
        '02 / Sellers with highest late rates</div>',
        unsafe_allow_html=True
    )

    if not view.empty:

        # Horizontal chart for readable seller labels
        top = view.head(15).iloc[::-1].copy()

        top["Seller"] = (
            top["seller_id"]
            .astype(str)
            .str.slice(0, 9)
            + "..."
        )

        st.bar_chart(
            top.set_index("Seller")["late_rate"],
            horizontal=True,
            color="#347ce2"
        )

        st.markdown("#### Seller detail report")

        display = view[
            ["seller_id", "orders", "late_rate", "state"]
        ].copy()

        display["late_rate"] = (
            display["late_rate"] * 100
        ).round(2)

        display = display.rename(columns={
            "seller_id": "Seller ID",
            "orders": "Orders",
            "late_rate": "Late rate (%)",
            "state": "State"
        })

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Late rate (%)": st.column_config.ProgressColumn(
                    "Late rate (%)",
                    min_value=0,
                    max_value=100,
                    format="%.2f%%"
                )
            }
        )

        st.download_button(
            "Export seller report (CSV)",
            data=display.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="smartship_seller_report.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "No sellers match these filters. "
            "Try reducing the minimum order count."
        )

    st.caption(
        "Seller rates are descriptive, not proof of "
        "seller fault. Multi-seller orders are counted "
        "once for each participating seller."
    )


# ==========================================
# 8. FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
        SmartShip AI · Machine Learning Final Project
        · Olist Brazilian E-Commerce Dataset
        · Delivery Risk Support Prototype
    </div>
    """,
    unsafe_allow_html=True
)
