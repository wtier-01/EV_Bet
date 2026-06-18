import streamlit as st
import pandas as pd

from ev_engine import (
    SPORTS,
    CORE_MARKETS,
    PROP_MARKETS,
    DEFAULT_TARGET_BOOKS,
    DEFAULT_REFERENCE_BOOKS,
    fetch_full_board,
    calculate_ev_opportunities,
    format_opportunities_for_display,
)


st.set_page_config(
    page_title="EV Bet Dashboard",
    page_icon="📈",
    layout="wide",
)


CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.metric-card {
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 14px;
    padding: 18px;
    background: rgba(250, 250, 250, 0.04);
}

.big-title {
    font-size: 2.1rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #777;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.bet-card {
    border: 1px solid rgba(49, 51, 63, 0.18);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 12px;
    background: rgba(250, 250, 250, 0.03);
}

.bet-title {
    font-size: 1.05rem;
    font-weight: 750;
}

.bet-meta {
    font-size: 0.9rem;
    color: #777;
    margin-top: 4px;
}

.ev-positive {
    font-size: 1.4rem;
    font-weight: 800;
}

.small-label {
    font-size: 0.8rem;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.warning-box {
    border: 1px solid rgba(255, 193, 7, 0.5);
    background: rgba(255, 193, 7, 0.08);
    padding: 12px 14px;
    border-radius: 12px;
}

.info-box {
    border: 1px solid rgba(0, 123, 255, 0.25);
    background: rgba(0, 123, 255, 0.06);
    padding: 12px 14px;
    border-radius: 12px;
    margin-bottom: 1rem;
}

.badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    background: rgba(255, 75, 75, 0.12);
    color: #ff4b4b;
    margin-left: 6px;
}

.subtle-divider {
    height: 1px;
    background: rgba(49, 51, 63, 0.12);
    margin: 8px 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_state():
    if "raw_odds" not in st.session_state:
        st.session_state.raw_odds = pd.DataFrame()

    if "ev_results" not in st.session_state:
        st.session_state.ev_results = pd.DataFrame()

    if "quota_log" not in st.session_state:
        st.session_state.quota_log = []

    if "last_fetch_status" not in st.session_state:
        st.session_state.last_fetch_status = ""


def sidebar_controls():
    st.sidebar.header("Controls")

    selected_sports = st.sidebar.multiselect(
        "Sports",
        options=list(SPORTS.keys()),
        default=["MLB"],
    )

    st.sidebar.divider()

    market_mode = st.sidebar.radio(
        "Market Mode",
        options=["Core markets only", "Props only", "Core + props"],
        index=0,
        help=(
            "Props cost more API calls because The Odds API fetches player props "
            "one event at a time."
        ),
    )

    include_core = market_mode in ["Core markets only", "Core + props"]
    include_props = market_mode in ["Props only", "Core + props"]

    selected_core_markets = []
    if include_core:
        selected_core_markets = st.sidebar.multiselect(
            "Core Markets",
            options=CORE_MARKETS,
            default=CORE_MARKETS,
            format_func=lambda x: {
                "h2h": "Moneyline",
                "spreads": "Spreads",
                "totals": "Totals",
            }.get(x, x),
        )

    selected_prop_markets = {}
    if include_props:
        st.sidebar.warning(
            "Props can use many API credits. Start small: 1 sport, 1-3 prop markets, 3-5 events."
        )

        max_prop_events = st.sidebar.slider(
            "Max Prop Events Per Sport",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
        )

        for sport in selected_sports:
            available_props = PROP_MARKETS.get(sport, [])

            if available_props:
                selected_prop_markets[sport] = st.sidebar.multiselect(
                    f"{sport} Prop Markets",
                    options=available_props,
                    default=available_props[:3],
                    format_func=lambda x: x.replace("_", " ").title(),
                )
            else:
                selected_prop_markets[sport] = []
    else:
        max_prop_events = 5

    st.sidebar.divider()

    st.sidebar.subheader("Sportsbooks")

    target_books_text = st.sidebar.text_area(
        "Your Books",
        value=", ".join(DEFAULT_TARGET_BOOKS),
        help="These are the books you can actually place bets on.",
    )

    reference_books_text = st.sidebar.text_area(
        "Reference Market Books",
        value=", ".join(DEFAULT_REFERENCE_BOOKS),
        help=(
            "These books create the no-vig market probability. Usually include your books "
            "plus additional market books."
        ),
    )

    target_books = clean_book_list(target_books_text)
    reference_books = clean_book_list(reference_books_text)

    st.sidebar.divider()

    st.sidebar.subheader("EV Settings")

    min_ev_percent = st.sidebar.slider(
        "Minimum Alpha EV %",
        min_value=-10.0,
        max_value=25.0,
        value=1.0,
        step=0.25,
        help=(
            "Filters on Alpha EV. Alpha EV uses Predictive EV when available, "
            "otherwise it uses Market EV."
        ),
    )

    min_reference_books = st.sidebar.slider(
        "Minimum Reference Books",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Higher is cleaner but may reduce the number of opportunities.",
    )

    bankroll = st.sidebar.number_input(
        "Bankroll",
        min_value=1.0,
        value=1000.0,
        step=100.0,
    )

    kelly_multiplier = st.sidebar.select_slider(
        "Kelly Multiplier",
        options=[0.05, 0.10, 0.20, 0.25, 0.33, 0.50, 1.00],
        value=0.25,
        help="Quarter Kelly is a conservative default.",
    )

    include_predictive_ev = st.sidebar.toggle(
        "Use Predictive EV When Available",
        value=True,
        help=(
            "Currently applies to MLB core markets. Other sports/props still use Market EV."
        ),
    )

    regions = st.sidebar.selectbox(
        "Region",
        options=["us", "us2", "uk", "eu", "au"],
        index=0,
    )

    return {
        "selected_sports": selected_sports,
        "include_core": include_core,
        "include_props": include_props,
        "selected_core_markets": selected_core_markets,
        "selected_prop_markets": selected_prop_markets,
        "max_prop_events": max_prop_events,
        "target_books": target_books,
        "reference_books": reference_books,
        "min_ev_percent": min_ev_percent,
        "min_reference_books": min_reference_books,
        "bankroll": bankroll,
        "kelly_multiplier": kelly_multiplier,
        "include_predictive_ev": include_predictive_ev,
        "regions": regions,
    }


def clean_book_list(text):
    return [
        item.strip().lower()
        for item in text.split(",")
        if item.strip()
    ]


@st.cache_data(ttl=60, show_spinner=False)
def cached_fetch_full_board(
    selected_sports,
    include_core,
    include_props,
    selected_core_markets,
    selected_prop_markets_tuple,
    regions,
    max_prop_events,
):
    selected_prop_markets = {
        sport: list(markets)
        for sport, markets in selected_prop_markets_tuple
    }

    return fetch_full_board(
        selected_sports=list(selected_sports),
        include_core=include_core,
        include_props=include_props,
        selected_core_markets=list(selected_core_markets),
        selected_prop_markets=selected_prop_markets,
        regions=regions,
        max_prop_events_per_sport=max_prop_events,
    )


def main():
    init_state()

    controls = sidebar_controls()

    st.markdown('<div class="big-title">EV Bet Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Full-market EV and predictive alpha scanner for your sportsbooks.</div>',
        unsafe_allow_html=True,
    )

    if not controls["selected_sports"]:
        st.info("Select at least one sport in the sidebar.")
        return

    top_left, top_mid, top_right = st.columns([1, 1, 1])

    with top_left:
        fetch_clicked = st.button(
            "Fetch Odds + Calculate EV",
            use_container_width=True,
            type="primary",
        )

    with top_mid:
        clear_cache_clicked = st.button(
            "Clear Cache",
            use_container_width=True,
        )

    with top_right:
        st.caption("Cache refreshes every 60 seconds by default.")

    if clear_cache_clicked:
        st.cache_data.clear()
        st.success("Cache cleared.")

    if fetch_clicked:
        with st.spinner("Fetching odds and calculating EV..."):
            try:
                selected_prop_markets_tuple = tuple(
                    (sport, tuple(markets))
                    for sport, markets in controls["selected_prop_markets"].items()
                )

                raw_odds, quota_log = cached_fetch_full_board(
                    selected_sports=tuple(controls["selected_sports"]),
                    include_core=controls["include_core"],
                    include_props=controls["include_props"],
                    selected_core_markets=tuple(controls["selected_core_markets"]),
                    selected_prop_markets_tuple=selected_prop_markets_tuple,
                    regions=controls["regions"],
                    max_prop_events=controls["max_prop_events"],
                )

                ev_results = calculate_ev_opportunities(
                    df=raw_odds,
                    target_books=controls["target_books"],
                    reference_books=controls["reference_books"],
                    min_reference_books=controls["min_reference_books"],
                    min_ev_percent=controls["min_ev_percent"],
                    bankroll=controls["bankroll"],
                    kelly_multiplier=controls["kelly_multiplier"],
                    include_predictive_ev=controls["include_predictive_ev"],
                )

                st.session_state.raw_odds = raw_odds
                st.session_state.ev_results = ev_results
                st.session_state.quota_log = quota_log
                st.session_state.last_fetch_status = "success"

            except Exception as exc:
                st.session_state.last_fetch_status = "error"
                st.error(str(exc))

    render_dashboard()


def render_dashboard():
    raw_odds = st.session_state.raw_odds
    ev_results = st.session_state.ev_results
    quota_log = st.session_state.quota_log

    if raw_odds.empty and ev_results.empty:
        st.markdown(
            """
            <div class="warning-box">
            <b>Start here:</b> choose your sports, markets, and books in the sidebar, then click
            <b>Fetch Odds + Calculate EV</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    with metric_1:
        st.metric("Raw Odds Rows", f"{len(raw_odds):,}")

    with metric_2:
        st.metric("EV Opportunities", f"{len(ev_results):,}")

    with metric_3:
        if not ev_results.empty and "alpha_ev_percent" in ev_results.columns:
            st.metric("Best Alpha EV", f"{ev_results['alpha_ev_percent'].max():.2f}%")
        else:
            st.metric("Best Alpha EV", "—")

    with metric_4:
        if not ev_results.empty and "alpha_ev_percent" in ev_results.columns:
            st.metric("Avg Alpha EV", f"{ev_results['alpha_ev_percent'].mean():.2f}%")
        else:
            st.metric("Avg Alpha EV", "—")

    st.divider()

    tab_best, tab_table, tab_raw, tab_quota, tab_explain = st.tabs(
        [
            "Best Bets",
            "Full EV Table",
            "Raw Odds",
            "API Usage",
            "How It Works",
        ]
    )

    with tab_best:
        render_best_bets(ev_results)

    with tab_table:
        render_full_table(ev_results)

    with tab_raw:
        render_raw_odds(raw_odds)

    with tab_quota:
        render_quota_log(quota_log)

    with tab_explain:
        render_explanation()


def render_best_bets(ev_results: pd.DataFrame):
    st.subheader("Best Bets to Place")

    if ev_results.empty:
        st.info(
            "No EV opportunities found with your current filters. Try lowering Minimum Alpha EV %, "
            "lowering Minimum Reference Books, adding more reference books, or checking whether the selected sport has active markets."
        )
        return

    top_n = st.slider(
        "Number of bets to show",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
    )

    sort_col = "alpha_ev_percent" if "alpha_ev_percent" in ev_results.columns else None

    if sort_col:
        top = ev_results.sort_values(by=sort_col, ascending=False).head(top_n).copy()
    else:
        top = ev_results.head(top_n).copy()

    for _, row in top.iterrows():
        alpha_ev = get_number(row, "alpha_ev_percent", 0.0)
        market_ev = get_number(row, "market_ev_percent", None)
        predictive_ev = get_number(row, "predictive_ev_percent", None)

        book = row.get("book", "")
        game = row.get("game", "")
        market = clean_market_name(row.get("market", row.get("market_key", "")))
        selection = row.get("selection", "")
        description = row.get("description", "")
        point = row.get("point", None)
        price = row.get("price", row.get("odds_american", None))

        alpha_fair_odds = row.get("alpha_fair_american_odds", None)
        market_fair_odds = row.get("fair_american_odds", None)
        predictive_fair_odds = row.get("predictive_fair_american_odds", None)

        suggested_bet = get_number(row, "suggested_bet_size", 0.0)
        market_edge = get_number(row, "market_edge_probability_points", None)

        alpha_probability = get_number(row, "alpha_probability", None)
        market_probability = get_number(row, "fair_probability", None)
        predictive_probability = get_number(row, "predictive_probability", None)
        push_probability = get_number(row, "push_probability", None)

        model_type = row.get("model_type", None)

        bet_text = build_readable_bet(market, selection, description, point)

        odds_text = format_american(price)
        alpha_fair_text = format_american(alpha_fair_odds)
        market_fair_text = format_american(market_fair_odds)
        predictive_fair_text = format_american(predictive_fair_odds)

        market_ev_text = format_percent(market_ev)
        predictive_ev_text = format_percent(predictive_ev)
        alpha_probability_text = format_probability(alpha_probability)
        market_probability_text = format_probability(market_probability)
        predictive_probability_text = format_probability(predictive_probability)
        push_probability_text = format_probability(push_probability)
        market_edge_text = "—" if market_edge is None else f"{market_edge:.2f} pts"

        if is_valid_value(model_type):
            signal_text = f"Predictive model: {model_type}"
            signal_badge = "Predictive"
        else:
            signal_text = "Signal: full-market no-vig consensus"
            signal_badge = "Market"

        col1, col2, col3 = st.columns([4.5, 1.35, 1.35])

        with col1:
            st.markdown(
                f"""
                <div class="bet-card">
                    <div class="bet-title">
                        {book} — {bet_text}
                        <span class="badge">{signal_badge}</span>
                    </div>
                    <div class="bet-meta">{game}</div>
                    <div class="subtle-divider"></div>
                    <div class="bet-meta">
                        Odds: <b>{odds_text}</b> | Alpha Fair Odds: <b>{alpha_fair_text}</b>
                    </div>
                    <div class="bet-meta">
                        Market EV: <b>{market_ev_text}</b> | Predictive EV: <b>{predictive_ev_text}</b>
                    </div>
                    <div class="bet-meta">
                        Market Fair: <b>{market_fair_text}</b> | Predictive Fair: <b>{predictive_fair_text}</b>
                    </div>
                    <div class="bet-meta">
                        Alpha Prob: <b>{alpha_probability_text}</b> | Market Prob: <b>{market_probability_text}</b> | Predictive Prob: <b>{predictive_probability_text}</b> | Push: <b>{push_probability_text}</b>
                    </div>
                    <div class="bet-meta">{signal_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown('<div class="small-label">Alpha EV</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="ev-positive">{alpha_ev:.2f}%</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Market edge: {market_edge_text}")

        with col3:
            st.markdown('<div class="small-label">Suggested</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="ev-positive">${suggested_bet:.2f}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Odds: {odds_text}")


def render_full_table(ev_results: pd.DataFrame):
    st.subheader("Full EV Table")

    if ev_results.empty:
        st.info("No EV rows to show.")
        return

    display = format_opportunities_for_display(ev_results)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download EV Table CSV",
        data=csv,
        file_name="ev_opportunities.csv",
        mime="text/csv",
    )


def render_raw_odds(raw_odds: pd.DataFrame):
    st.subheader("Raw Odds Pulled From API")

    if raw_odds.empty:
        st.info("No raw odds loaded.")
        return

    st.dataframe(
        raw_odds,
        use_container_width=True,
        hide_index=True,
    )


def render_quota_log(quota_log):
    st.subheader("API Usage / Quota Log")

    if not quota_log:
        st.info("No quota data yet.")
        return

    quota_df = pd.DataFrame(quota_log)
    st.dataframe(
        quota_df,
        use_container_width=True,
        hide_index=True,
    )


def render_explanation():
    st.subheader("How This Dashboard Calculates EV")

    st.markdown(
        """
        This dashboard now separates **Market EV**, **Predictive EV**, and **Alpha EV**.

        ### 1. Market EV

        Market EV compares your book's price against the full-market no-vig consensus.

        The app:

        - Pulls odds from your selected sports and markets.
        - Uses your Reference Market Books to estimate the fair market price.
        - Removes the vig inside each book's two-sided market.
        - Averages those no-vig prices across the reference market.
        - Compares your available book price against that fair probability.

        This is the cleanest baseline for line shopping.

        ### 2. Predictive EV

        Predictive EV is the alpha layer.

        For MLB core markets, the app uses a market-calibrated Poisson model:

        - Moneyline market estimates team win probability.
        - Totals market estimates the scoring environment.
        - The model builds a score distribution.
        - That score distribution prices moneylines, spreads, and totals.
        - The app compares your book's price against the model probability.

        Currently, Predictive EV applies to:

        - MLB moneylines
        - MLB spreads
        - MLB totals

        Other sports and player props still use Market EV until predictive models are added for them.

        ### 3. Alpha EV

        Alpha EV is the main ranking column.

        The rule is:

        `Alpha EV = Predictive EV when available, otherwise Market EV`

        So the Best Bets tab ranks by the strongest available signal.

        ### 4. Suggested bet sizing

        Suggested sizing uses fractional Kelly.

        Quarter Kelly is the default because full Kelly can be too aggressive.

        ### Important

        This is not a guarantee of profit. It is a pricing and signal-discovery tool. The goal is to find bets where your available price appears better than either the full-market consensus or the predictive model.
        """
    )


def clean_market_name(market):
    mapping = {
        "h2h": "Moneyline",
        "spreads": "Spread",
        "totals": "Total",
    }

    if market in mapping:
        return mapping[market]

    if not market:
        return ""

    return str(market).replace("_", " ").title()


def build_readable_bet(market, selection, description, point):
    pieces = []

    if is_valid_value(description):
        pieces.append(str(description))

    if is_valid_value(selection):
        pieces.append(str(selection))

    if is_valid_value(point):
        pieces.append(str(point))

    bet = " ".join(pieces).strip()

    if not bet:
        bet = str(selection)

    return f"{market}: {bet}"


def format_american(value):
    if not is_valid_value(value):
        return "—"

    try:
        value = int(round(float(value)))
    except Exception:
        return "—"

    if value > 0:
        return f"+{value}"

    return str(value)


def format_percent(value):
    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.2f}%"
    except Exception:
        return "—"


def format_probability(value):
    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.2%}"
    except Exception:
        return "—"


def get_number(row, column, default=None):
    try:
        value = row.get(column, default)

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def is_valid_value(value):
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    if isinstance(value, str) and value.strip() == "":
        return False

    return True


if __name__ == "__main__":
    main()