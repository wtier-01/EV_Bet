from __future__ import annotations

import os
import math
import time
import requests
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple


load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"

SPORTS = {
    "MLB": "baseball_mlb",
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "College Football": "americanfootball_ncaaf",
}

CORE_MARKETS = ["h2h", "spreads", "totals"]

PROP_MARKETS = {
    "MLB": [
        "batter_home_runs",
        "batter_hits",
        "batter_total_bases",
        "batter_rbis",
        "batter_runs_scored",
        "batter_hits_runs_rbis",
        "pitcher_strikeouts",
        "pitcher_hits_allowed",
        "pitcher_walks",
        "pitcher_earned_runs",
        "pitcher_outs",
    ],
    "NFL": [
        "player_pass_tds",
        "player_pass_yds",
        "player_pass_completions",
        "player_pass_attempts",
        "player_pass_interceptions",
        "player_rush_yds",
        "player_rush_attempts",
        "player_receptions",
        "player_reception_yds",
        "player_anytime_td",
        "player_kicking_points",
    ],
    "NBA": [
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_threes",
        "player_blocks",
        "player_steals",
        "player_turnovers",
        "player_points_rebounds_assists",
        "player_points_rebounds",
        "player_points_assists",
        "player_rebounds_assists",
    ],
    "College Football": [
        "player_pass_tds",
        "player_pass_yds",
        "player_rush_yds",
        "player_receptions",
        "player_reception_yds",
        "player_anytime_td",
    ],
}

DEFAULT_TARGET_BOOKS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "betrivers",
    "espnbet",
    "fanatics",
]

DEFAULT_REFERENCE_BOOKS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "betrivers",
    "espnbet",
    "fanatics",
    "betonlineag",
    "bovada",
    "lowvig",
    "mybookieag",
    "betus",
]


# -----------------------------
# Basic odds helpers
# -----------------------------

def require_api_key() -> None:
    if not ODDS_API_KEY or ODDS_API_KEY == "PASTE_YOUR_ODDS_API_KEY_HERE":
        raise RuntimeError(
            "Missing ODDS_API_KEY. Add your key to .env like this: "
            "ODDS_API_KEY=your_key_here"
        )


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def american_to_decimal(american_odds: Any) -> Optional[float]:
    odds = safe_float(american_odds)
    if odds is None:
        return None
    if odds > 0:
        return 1 + odds / 100
    if odds < 0:
        return 1 + 100 / abs(odds)
    return None


def american_to_implied_prob(american_odds: Any) -> Optional[float]:
    odds = safe_float(american_odds)
    if odds is None:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return None


def decimal_to_american(decimal_odds: float) -> Optional[int]:
    if decimal_odds is None or decimal_odds <= 1:
        return None
    if decimal_odds >= 2:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def probability_to_american(probability: Optional[float]) -> Optional[int]:
    if probability is None or probability <= 0 or probability >= 1:
        return None
    return decimal_to_american(1 / probability)


def remove_vig(probs: List[float]) -> List[float]:
    total = sum(p for p in probs if p is not None)
    if total <= 0:
        return []
    return [p / total for p in probs]


def ev_no_push(win_probability: float, american_odds: Any) -> Optional[float]:
    decimal_odds = american_to_decimal(american_odds)
    if decimal_odds is None:
        return None

    profit_if_win = decimal_odds - 1
    loss_probability = 1 - win_probability

    return (win_probability * profit_if_win) - loss_probability


def ev_with_push(
    win_probability: float,
    push_probability: float,
    american_odds: Any,
) -> Optional[float]:
    decimal_odds = american_to_decimal(american_odds)
    if decimal_odds is None:
        return None

    profit_if_win = decimal_odds - 1
    loss_probability = max(0.0, 1.0 - win_probability - push_probability)

    return (win_probability * profit_if_win) - loss_probability


def kelly_fraction(win_probability: float, american_odds: Any) -> Optional[float]:
    decimal_odds = american_to_decimal(american_odds)
    if decimal_odds is None:
        return None

    b = decimal_odds - 1
    p = win_probability
    q = 1 - p

    if b <= 0:
        return None

    return max(0.0, ((b * p) - q) / b)


# -----------------------------
# Reliability / alpha quality helpers
# -----------------------------

def calculate_blended_ev_percent(row: pd.Series, predictive_weight: float = 0.65) -> float:
    """
    Blends predictive EV with market EV so the model does not blindly override the market.

    If predictive EV exists:
        blended EV = predictive_weight * predictive EV + market_weight * market EV

    If predictive EV does not exist:
        blended EV = market EV
    """
    market_ev = safe_float(row.get("market_ev_percent"))
    predictive_ev = safe_float(row.get("predictive_ev_percent"))

    if predictive_ev is None:
        return market_ev if market_ev is not None else 0.0

    if market_ev is None:
        return predictive_ev

    market_weight = 1 - predictive_weight
    return (predictive_weight * predictive_ev) + (market_weight * market_ev)


def calculate_blended_probability(row: pd.Series, predictive_weight: float = 0.65) -> Optional[float]:
    """
    Conservative probability used for sizing.

    This keeps Kelly sizing from using a pure predictive probability when the market disagrees.
    """
    market_probability = safe_float(row.get("fair_probability"))
    predictive_probability = safe_float(row.get("predictive_probability"))

    if predictive_probability is None:
        return market_probability

    if market_probability is None:
        return predictive_probability

    market_weight = 1 - predictive_weight
    blended = (predictive_weight * predictive_probability) + (market_weight * market_probability)

    return max(0.0001, min(0.9999, blended))


def build_warning_flags(row: pd.Series) -> str:
    flags = []

    market_ev = safe_float(row.get("market_ev_percent"))
    predictive_ev = safe_float(row.get("predictive_ev_percent"))
    reference_count = safe_float(row.get("reference_book_count"))
    market_edge = safe_float(row.get("market_edge_probability_points"))

    if reference_count is not None and reference_count < 3:
        flags.append("low reference count")

    if market_ev is not None and market_ev < -7.5:
        flags.append("market strongly disagrees")

    if market_ev is not None and market_ev < -15:
        flags.append("high negative market EV")

    if market_edge is not None and market_edge < -5:
        flags.append("negative market edge")

    if market_ev is not None and predictive_ev is not None:
        disagreement = abs(predictive_ev - market_ev)

        if disagreement >= 20:
            flags.append("extreme model/market gap")
        elif disagreement >= 15:
            flags.append("large model/market gap")

    if predictive_ev is None:
        flags.append("market EV only")

    return ", ".join(flags)


def assign_bet_tier(row: pd.Series) -> str:
    blended_ev = safe_float(row.get("blended_ev_percent"))
    market_ev = safe_float(row.get("market_ev_percent"))
    predictive_ev = safe_float(row.get("predictive_ev_percent"))
    reference_count = safe_float(row.get("reference_book_count"))

    blended_ev = blended_ev if blended_ev is not None else 0.0
    market_ev = market_ev if market_ev is not None else 0.0
    reference_count = reference_count if reference_count is not None else 0

    has_predictive = predictive_ev is not None

    # Model likes it, but the broader market strongly disagrees.
    if has_predictive and predictive_ev > 0 and market_ev < -7.5:
        return "Review Only"

    # Best case: predictive edge with reasonable market confirmation.
    if has_predictive:
        if blended_ev >= 2.0 and market_ev >= -2.5 and reference_count >= 4:
            return "Strong Bet"

        if blended_ev >= 1.0 and market_ev >= -7.5 and reference_count >= 3:
            return "Model Lean"

    # Non-predictive markets can still be useful market EV opportunities.
    if not has_predictive:
        if market_ev >= 1.0 and reference_count >= 3:
            return "Market EV Bet"

    return "Avoid"


def calculate_bet_quality_score(row: pd.Series) -> float:
    blended_ev = safe_float(row.get("blended_ev_percent")) or 0.0
    market_ev = safe_float(row.get("market_ev_percent")) or 0.0
    predictive_ev = safe_float(row.get("predictive_ev_percent"))
    reference_count = safe_float(row.get("reference_book_count")) or 0
    bet_tier = row.get("bet_tier", "Avoid")

    score = 50.0

    # Reward blended EV, but cap impact so one huge model output cannot dominate.
    score += min(max(blended_ev * 3.0, -30), 30)

    # Reward or punish market confirmation.
    if market_ev >= 2:
        score += min(market_ev * 2.0, 18)
    elif market_ev >= 0:
        score += 10
    elif market_ev >= -2.5:
        score += 5
    elif market_ev < -15:
        score -= 35
    elif market_ev < -7.5:
        score -= 20

    # Reward reference market depth.
    score += min(reference_count * 2.0, 12)

    # Reward model/market agreement. Punish major disagreement.
    if predictive_ev is not None:
        disagreement = abs(predictive_ev - market_ev)

        if disagreement <= 5:
            score += 10
        elif disagreement <= 10:
            score += 4
        elif disagreement >= 20:
            score -= 20
        elif disagreement >= 15:
            score -= 12

    # Tier-specific adjustment.
    if bet_tier == "Strong Bet":
        score += 10
    elif bet_tier == "Market EV Bet":
        score += 6
    elif bet_tier == "Model Lean":
        score += 2
    elif bet_tier == "Review Only":
        score -= 12
    elif bet_tier == "Avoid":
        score -= 30

    return round(max(0, min(100, score)), 1)


def get_tier_rank(tier: str) -> int:
    tier_rank = {
        "Strong Bet": 1,
        "Market EV Bet": 2,
        "Model Lean": 3,
        "Review Only": 4,
        "Avoid": 5,
    }

    return tier_rank.get(tier, 9)


# -----------------------------
# API fetchers
# -----------------------------

def api_get(path: str, params: Dict) -> Tuple[dict | list, Dict]:
    require_api_key()

    params = dict(params)
    params["apiKey"] = ODDS_API_KEY

    response = requests.get(
        f"{BASE_URL}{path}",
        params=params,
        timeout=30,
    )

    quota = {
        "requests_remaining": response.headers.get("x-requests-remaining"),
        "requests_used": response.headers.get("x-requests-used"),
        "requests_last": response.headers.get("x-requests-last"),
    }

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text[:500]}")

    return response.json(), quota


def fetch_core_odds(
    sport_key: str,
    markets: List[str],
    regions: str = "us",
) -> Tuple[List[dict], Dict]:
    data, quota = api_get(
        f"/sports/{sport_key}/odds",
        {
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
    )
    return data, quota


def fetch_events(sport_key: str) -> Tuple[List[dict], Dict]:
    data, quota = api_get(
        f"/sports/{sport_key}/events",
        {
            "dateFormat": "iso",
        },
    )
    return data, quota


def fetch_event_props(
    sport_key: str,
    event_id: str,
    markets: List[str],
    regions: str = "us",
) -> Tuple[dict, Dict]:
    data, quota = api_get(
        f"/sports/{sport_key}/events/{event_id}/odds",
        {
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
    )
    return data, quota


# -----------------------------
# Flattening
# -----------------------------

def flatten_odds(events: List[dict], sport_label: str) -> pd.DataFrame:
    rows = []

    for event in events:
        event_id = event.get("id")
        commence_time = event.get("commence_time")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        sport_key = SPORTS.get(sport_label)

        for book in event.get("bookmakers", []):
            book_key = book.get("key")
            book_title = book.get("title")
            last_update = book.get("last_update")

            for market in book.get("markets", []):
                market_key = market.get("key")

                for outcome in market.get("outcomes", []):
                    selection = outcome.get("name")
                    description = outcome.get("description")
                    price = outcome.get("price")
                    point = outcome.get("point")

                    rows.append(
                        {
                            "sport": sport_label,
                            "sport_key": sport_key,
                            "event_id": event_id,
                            "commence_time": commence_time,
                            "home_team": home_team,
                            "away_team": away_team,
                            "game": f"{away_team} @ {home_team}",
                            "bookmaker_key": book_key,
                            "book_key": book_key,
                            "book": book_title,
                            "market_key": market_key,
                            "market": market_key,
                            "selection": selection,
                            "description": description,
                            "point": point,
                            "odds_american": price,
                            "price": price,
                            "implied_probability": american_to_implied_prob(price),
                            "implied_prob": american_to_implied_prob(price),
                            "last_update": last_update,
                        }
                    )

    return pd.DataFrame(rows)


# -----------------------------
# Correct market grouping
# -----------------------------

def point_group(row: pd.Series | Dict[str, Any]) -> Optional[float]:
    market = row.get("market_key") or row.get("market")
    point = safe_float(row.get("point"))

    if market == "h2h":
        return None

    if point is None:
        return None

    if market == "spreads":
        return abs(round(point, 3))

    return round(point, 3)


def description_group(row: pd.Series | Dict[str, Any]) -> str:
    market = row.get("market_key") or row.get("market")

    # Player props need description/player grouped.
    # Core markets generally do not.
    if market in {"h2h", "spreads", "totals"}:
        return ""

    desc = row.get("description")
    if desc is None or pd.isna(desc):
        return ""

    return str(desc)


def market_group_key(row: pd.Series | Dict[str, Any]) -> Tuple:
    return (
        row.get("sport_key"),
        row.get("event_id"),
        row.get("bookmaker_key") or row.get("book_key"),
        row.get("market_key") or row.get("market"),
        point_group(row),
        description_group(row),
    )


def consensus_group_key(row: pd.Series | Dict[str, Any]) -> Tuple:
    return (
        row.get("sport_key"),
        row.get("event_id"),
        row.get("market_key") or row.get("market"),
        point_group(row),
        description_group(row),
    )


def selection_key(row: pd.Series | Dict[str, Any]) -> Tuple:
    return (
        consensus_group_key(row),
        row.get("selection"),
    )


def build_consensus_probabilities(
    df: pd.DataFrame,
    reference_books: List[str],
    min_reference_books: int = 2,
) -> pd.DataFrame:
    """
    Correct approach:
    1. For each book, identify a full two-sided market.
    2. Remove vig inside that one book's market.
    3. Average no-vig probabilities across all reference books.
    """

    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    reference_books = {b.lower() for b in reference_books}

    work = work[
        work["bookmaker_key"].str.lower().isin(reference_books)
        & work["implied_probability"].notna()
    ].copy()

    if work.empty:
        return pd.DataFrame()

    work["book_market_group"] = work.apply(market_group_key, axis=1)
    work["consensus_market_group"] = work.apply(consensus_group_key, axis=1)
    work["selection_group"] = work.apply(selection_key, axis=1)

    samples = []

    for _, group in work.groupby("book_market_group"):
        # Most normal markets are two-way.
        # We skip broken one-sided markets to avoid 100% fair probability bugs.
        if len(group) != 2:
            continue

        if group["selection"].nunique() != 2:
            continue

        probs = group["implied_probability"].tolist()
        fair_probs = remove_vig(probs)

        if len(fair_probs) != len(group):
            continue

        for (_, row), fair_prob in zip(group.iterrows(), fair_probs):
            samples.append(
                {
                    "consensus_market_group": row["consensus_market_group"],
                    "selection": row["selection"],
                    "fair_probability_sample": fair_prob,
                    "source_book": row["bookmaker_key"],
                }
            )

    if not samples:
        return pd.DataFrame()

    sample_df = pd.DataFrame(samples)

    consensus = (
        sample_df
        .groupby(["consensus_market_group", "selection"], as_index=False)
        .agg(
            fair_probability=("fair_probability_sample", "mean"),
            reference_book_count=("source_book", "nunique"),
        )
    )

    consensus = consensus[
        consensus["reference_book_count"] >= min_reference_books
    ].copy()

    consensus["fair_american_odds"] = consensus["fair_probability"].apply(
        probability_to_american
    )

    return consensus


# -----------------------------
# MLB predictive model
# -----------------------------

def poisson_pmf(lam: float, max_runs: int = 30) -> List[float]:
    probs = []

    for k in range(max_runs + 1):
        probs.append(math.exp(-lam) * (lam ** k) / math.factorial(k))

    total = sum(probs)

    if total > 0:
        probs = [p / total for p in probs]

    return probs


def score_distribution(
    away_lambda: float,
    home_lambda: float,
    max_runs: int = 30,
) -> Dict[Tuple[int, int], float]:
    away_probs = poisson_pmf(away_lambda, max_runs=max_runs)
    home_probs = poisson_pmf(home_lambda, max_runs=max_runs)

    dist = {}

    for away_runs, away_prob in enumerate(away_probs):
        for home_runs, home_prob in enumerate(home_probs):
            dist[(away_runs, home_runs)] = away_prob * home_prob

    return dist


def prob_total_side(
    total_lambda: float,
    line: float,
    side: str,
    max_runs: int = 35,
) -> Tuple[float, float]:
    probs = poisson_pmf(total_lambda, max_runs=max_runs)

    win = 0.0
    push = 0.0

    for total_runs, prob in enumerate(probs):
        if side.lower() == "over":
            if total_runs > line:
                win += prob
            elif total_runs == line:
                push += prob
        else:
            if total_runs < line:
                win += prob
            elif total_runs == line:
                push += prob

    return win, push


def infer_total_lambda_from_market(
    total_line: float,
    fair_over_probability: Optional[float],
) -> float:
    if fair_over_probability is None:
        return max(total_line, 6.0)

    low = 3.0
    high = 16.0

    for _ in range(60):
        mid = (low + high) / 2

        over_win, _ = prob_total_side(mid, total_line, "Over")
        under_win, _ = prob_total_side(mid, total_line, "Under")

        no_push_total = over_win + under_win

        if no_push_total <= 0:
            modeled_over = 0.5
        else:
            modeled_over = over_win / no_push_total

        if modeled_over < fair_over_probability:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def away_win_probability_from_lambdas(
    away_lambda: float,
    home_lambda: float,
) -> float:
    dist = score_distribution(away_lambda, home_lambda)

    away_win = 0.0
    tie = 0.0

    for (away_runs, home_runs), prob in dist.items():
        if away_runs > home_runs:
            away_win += prob
        elif away_runs == home_runs:
            tie += prob

    # Approximate extras as 50/50 from tied regulation.
    return away_win + 0.5 * tie


def solve_team_lambdas(
    total_lambda: float,
    target_away_win_probability: float,
) -> Tuple[float, float]:
    low = -6.0
    high = 6.0

    for _ in range(70):
        diff = (low + high) / 2

        away_lambda = max(0.1, (total_lambda + diff) / 2)
        home_lambda = max(0.1, total_lambda - away_lambda)

        modeled_away_win = away_win_probability_from_lambdas(
            away_lambda=away_lambda,
            home_lambda=home_lambda,
        )

        if modeled_away_win < target_away_win_probability:
            low = diff
        else:
            high = diff

    diff = (low + high) / 2
    away_lambda = max(0.1, (total_lambda + diff) / 2)
    home_lambda = max(0.1, total_lambda - away_lambda)

    return away_lambda, home_lambda


def cover_probability(
    dist: Dict[Tuple[int, int], float],
    selection: str,
    point: float,
    away_team: str,
    home_team: str,
) -> Tuple[float, float]:
    win = 0.0
    push = 0.0

    for (away_runs, home_runs), prob in dist.items():
        if selection == away_team:
            margin = away_runs - home_runs
        elif selection == home_team:
            margin = home_runs - away_runs
        else:
            continue

        result = margin + point

        if result > 0:
            win += prob
        elif result == 0:
            push += prob

    return win, push


def moneyline_probability(
    dist: Dict[Tuple[int, int], float],
    selection: str,
    away_team: str,
    home_team: str,
) -> Tuple[float, float]:
    win = 0.0
    tie = 0.0

    for (away_runs, home_runs), prob in dist.items():
        if away_runs == home_runs:
            tie += prob
            continue

        if selection == away_team and away_runs > home_runs:
            win += prob

        if selection == home_team and home_runs > away_runs:
            win += prob

    return win + 0.5 * tie, 0.0


def build_mlb_predictive_models(
    df: pd.DataFrame,
    reference_books: List[str],
    min_reference_books: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """
    Market-calibrated MLB model.

    Uses:
    - Full market moneyline to infer win probability
    - Full market total to infer scoring environment
    - Poisson score distribution to price ML, spreads, totals
    """

    if df.empty:
        return {}

    mlb = df[
        (df["sport_key"] == "baseball_mlb")
        & (df["market_key"].isin(["h2h", "spreads", "totals"]))
    ].copy()

    if mlb.empty:
        return {}

    consensus = build_consensus_probabilities(
        mlb,
        reference_books=reference_books,
        min_reference_books=min_reference_books,
    )

    if consensus.empty:
        return {}

    probability_lookup = {}

    for _, row in consensus.iterrows():
        group = row["consensus_market_group"]
        selection = row["selection"]
        probability_lookup[(group, selection)] = row["fair_probability"]

    models = {}

    for event_id, event_rows in mlb.groupby("event_id"):
        sample = event_rows.iloc[0]

        away_team = sample["away_team"]
        home_team = sample["home_team"]

        away_group = (
            "baseball_mlb",
            event_id,
            "h2h",
            None,
            "",
        )

        away_ml = probability_lookup.get((away_group, away_team))
        home_ml = probability_lookup.get((away_group, home_team))

        if away_ml is None or home_ml is None:
            continue

        ml_total = away_ml + home_ml
        if ml_total <= 0:
            continue

        target_away_win_probability = away_ml / ml_total

        totals = event_rows[event_rows["market_key"] == "totals"].copy()

        if totals.empty:
            continue

        totals["point_group"] = totals.apply(point_group, axis=1)

        main_total = (
            totals["point_group"]
            .dropna()
            .value_counts()
            .index
        )

        if len(main_total) == 0:
            continue

        total_line = float(main_total[0])

        total_group = (
            "baseball_mlb",
            event_id,
            "totals",
            total_line,
            "",
        )

        fair_over_probability = probability_lookup.get((total_group, "Over"))

        total_lambda = infer_total_lambda_from_market(
            total_line=total_line,
            fair_over_probability=fair_over_probability,
        )

        away_lambda, home_lambda = solve_team_lambdas(
            total_lambda=total_lambda,
            target_away_win_probability=target_away_win_probability,
        )

        dist = score_distribution(
            away_lambda=away_lambda,
            home_lambda=home_lambda,
        )

        models[event_id] = {
            "event_id": event_id,
            "away_team": away_team,
            "home_team": home_team,
            "target_away_win_probability": target_away_win_probability,
            "total_line": total_line,
            "total_lambda": total_lambda,
            "away_lambda": away_lambda,
            "home_lambda": home_lambda,
            "score_distribution": dist,
            "model_type": "market_calibrated_poisson",
        }

    return models


def estimate_mlb_predictive_probability(
    row: pd.Series,
    model: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float]]:
    market = row.get("market_key")
    selection = row.get("selection")
    point = safe_float(row.get("point"))

    away_team = model["away_team"]
    home_team = model["home_team"]
    dist = model["score_distribution"]

    if market == "h2h":
        return moneyline_probability(
            dist=dist,
            selection=selection,
            away_team=away_team,
            home_team=home_team,
        )

    if market == "spreads":
        if point is None:
            return None, None

        return cover_probability(
            dist=dist,
            selection=selection,
            point=point,
            away_team=away_team,
            home_team=home_team,
        )

    if market == "totals":
        if point is None:
            return None, None

        return prob_total_side(
            total_lambda=model["total_lambda"],
            line=point,
            side=selection,
        )

    return None, None


# -----------------------------
# Main EV calculation
# -----------------------------

def calculate_ev_opportunities(
    df: pd.DataFrame,
    target_books: List[str],
    reference_books: List[str],
    min_reference_books: int = 2,
    min_ev_percent: float = 0.0,
    bankroll: float = 1000,
    kelly_multiplier: float = 0.25,
    include_predictive_ev: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()

    target_books = {b.lower() for b in target_books}
    reference_books = [b.lower() for b in reference_books]

    consensus = build_consensus_probabilities(
        work,
        reference_books=reference_books,
        min_reference_books=min_reference_books,
    )

    if consensus.empty:
        return pd.DataFrame()

    work["consensus_market_group"] = work.apply(consensus_group_key, axis=1)

    target = work[
        work["bookmaker_key"].str.lower().isin(target_books)
        & work["price"].notna()
    ].copy()

    if target.empty:
        return pd.DataFrame()

    merged = target.merge(
        consensus,
        on=["consensus_market_group", "selection"],
        how="left",
    )

    merged = merged[merged["fair_probability"].notna()].copy()

    if merged.empty:
        return pd.DataFrame()

    merged["market_ev"] = merged.apply(
        lambda row: ev_no_push(row["fair_probability"], row["price"]),
        axis=1,
    )

    merged["market_ev_percent"] = merged["market_ev"] * 100
    merged["market_edge_probability_points"] = (
        merged["fair_probability"] - merged["implied_probability"]
    ) * 100

    merged["predictive_probability"] = None
    merged["push_probability"] = 0.0
    merged["predictive_ev"] = None
    merged["predictive_ev_percent"] = None
    merged["predictive_fair_american_odds"] = None
    merged["model_type"] = None
    merged["model_away_lambda"] = None
    merged["model_home_lambda"] = None
    merged["model_total_lambda"] = None
    merged["model_total_line"] = None

    if include_predictive_ev:
        models = build_mlb_predictive_models(
            work,
            reference_books=reference_books,
            min_reference_books=min_reference_books,
        )

        for idx, row in merged.iterrows():
            if row.get("sport_key") != "baseball_mlb":
                continue

            if row.get("market_key") not in {"h2h", "spreads", "totals"}:
                continue

            model = models.get(row.get("event_id"))

            if not model:
                continue

            win_prob, push_prob = estimate_mlb_predictive_probability(row, model)

            if win_prob is None or push_prob is None:
                continue

            predictive_ev = ev_with_push(
                win_probability=win_prob,
                push_probability=push_prob,
                american_odds=row["price"],
            )

            merged.at[idx, "predictive_probability"] = win_prob
            merged.at[idx, "push_probability"] = push_prob
            merged.at[idx, "predictive_ev"] = predictive_ev
            merged.at[idx, "predictive_ev_percent"] = predictive_ev * 100
            merged.at[idx, "predictive_fair_american_odds"] = probability_to_american(win_prob)
            merged.at[idx, "model_type"] = model["model_type"]
            merged.at[idx, "model_away_lambda"] = round(model["away_lambda"], 3)
            merged.at[idx, "model_home_lambda"] = round(model["home_lambda"], 3)
            merged.at[idx, "model_total_lambda"] = round(model["total_lambda"], 3)
            merged.at[idx, "model_total_line"] = model["total_line"]

    # Reliability layer:
    # Alpha EV now uses blended EV rather than raw predictive EV.
    # This keeps predictive signals useful while penalizing bets the market strongly disagrees with.
    merged["blended_ev_percent"] = merged.apply(
        lambda row: calculate_blended_ev_percent(row, predictive_weight=0.65),
        axis=1,
    )

    merged["blended_probability"] = merged.apply(
        lambda row: calculate_blended_probability(row, predictive_weight=0.65),
        axis=1,
    )

    merged["blended_fair_american_odds"] = merged["blended_probability"].apply(
        probability_to_american
    )

    merged["alpha_ev_percent"] = merged["blended_ev_percent"]

    merged["alpha_probability"] = merged["blended_probability"].where(
        merged["blended_probability"].notna(),
        merged["fair_probability"],
    )

    merged["alpha_fair_american_odds"] = merged["blended_fair_american_odds"].where(
        merged["blended_fair_american_odds"].notna(),
        merged["fair_american_odds"],
    )

    merged["warning_flags"] = merged.apply(build_warning_flags, axis=1)
    merged["bet_tier"] = merged.apply(assign_bet_tier, axis=1)
    merged["bet_quality_score"] = merged.apply(calculate_bet_quality_score, axis=1)
    merged["tier_rank"] = merged["bet_tier"].apply(get_tier_rank)

    merged["kelly_fraction"] = merged.apply(
        lambda row: kelly_fraction(row["alpha_probability"], row["price"]),
        axis=1,
    )

    merged["suggested_bet_size"] = (
        bankroll * merged["kelly_fraction"].fillna(0) * kelly_multiplier
    ).clip(lower=0)

    # Default filter keeps the app actionable while still allowing Review Only rows.
    # Avoid rows are dropped because they fail the reliability layer.
    merged = merged[
        merged["alpha_ev_percent"].notna()
        & (merged["alpha_ev_percent"] >= min_ev_percent)
        & (merged["bet_tier"] != "Avoid")
    ].copy()

    merged["commence_time"] = pd.to_datetime(
        merged["commence_time"],
        errors="coerce",
        utc=True,
    )

    merged = merged.sort_values(
        by=["tier_rank", "bet_quality_score", "alpha_ev_percent", "market_ev_percent"],
        ascending=[True, False, False, False],
    )

    cols = [
        "sport",
        "sport_key",
        "commence_time",
        "game",
        "home_team",
        "away_team",
        "book",
        "bookmaker_key",
        "book_key",
        "market",
        "market_key",
        "selection",
        "description",
        "point",
        "price",
        "odds_american",
        "fair_probability",
        "fair_american_odds",
        "market_ev_percent",
        "market_edge_probability_points",
        "predictive_probability",
        "push_probability",
        "predictive_ev_percent",
        "predictive_fair_american_odds",
        "blended_ev_percent",
        "blended_probability",
        "blended_fair_american_odds",
        "alpha_ev_percent",
        "alpha_probability",
        "alpha_fair_american_odds",
        "bet_quality_score",
        "bet_tier",
        "warning_flags",
        "tier_rank",
        "reference_book_count",
        "suggested_bet_size",
        "model_type",
        "model_away_lambda",
        "model_home_lambda",
        "model_total_lambda",
        "model_total_line",
        "last_update",
    ]

    return merged[[c for c in cols if c in merged.columns]].reset_index(drop=True)


# -----------------------------
# Full board fetcher
# -----------------------------

def fetch_full_board(
    selected_sports: List[str],
    include_core: bool = True,
    include_props: bool = False,
    selected_core_markets: Optional[List[str]] = None,
    selected_prop_markets: Optional[Dict[str, List[str]]] = None,
    regions: str = "us",
    max_prop_events_per_sport: int = 5,
    progress_callback=None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    all_frames = []
    quota_log = []

    selected_core_markets = selected_core_markets or CORE_MARKETS
    selected_prop_markets = selected_prop_markets or {}

    for sport_label in selected_sports:
        sport_key = SPORTS[sport_label]

        if include_core:
            if progress_callback:
                progress_callback(f"Fetching core markets for {sport_label}...")

            try:
                events, quota = fetch_core_odds(
                    sport_key=sport_key,
                    markets=selected_core_markets,
                    regions=regions,
                )

                quota_log.append(
                    {
                        "sport": sport_label,
                        "type": "core",
                        **quota,
                    }
                )

                frame = flatten_odds(events, sport_label)

                if not frame.empty:
                    all_frames.append(frame)

            except Exception as exc:
                quota_log.append(
                    {
                        "sport": sport_label,
                        "type": "core",
                        "error": str(exc),
                    }
                )

        if include_props:
            prop_markets = selected_prop_markets.get(sport_label, [])

            if prop_markets:
                try:
                    events, quota = fetch_events(sport_key)

                    quota_log.append(
                        {
                            "sport": sport_label,
                            "type": "events",
                            **quota,
                        }
                    )

                    events = events[:max_prop_events_per_sport]

                    for i, event in enumerate(events, start=1):
                        event_id = event.get("id")

                        if not event_id:
                            continue

                        if progress_callback:
                            progress_callback(
                                f"Fetching props for {sport_label} event {i}/{len(events)}..."
                            )

                        try:
                            event_odds, quota = fetch_event_props(
                                sport_key=sport_key,
                                event_id=event_id,
                                markets=prop_markets,
                                regions=regions,
                            )

                            quota_log.append(
                                {
                                    "sport": sport_label,
                                    "type": "props",
                                    "event_id": event_id,
                                    **quota,
                                }
                            )

                            frame = flatten_odds([event_odds], sport_label)

                            if not frame.empty:
                                all_frames.append(frame)

                            time.sleep(0.15)

                        except Exception as exc:
                            quota_log.append(
                                {
                                    "sport": sport_label,
                                    "type": "props",
                                    "event_id": event_id,
                                    "error": str(exc),
                                }
                            )

                except Exception as exc:
                    quota_log.append(
                        {
                            "sport": sport_label,
                            "type": "events",
                            "error": str(exc),
                        }
                    )

    if not all_frames:
        return pd.DataFrame(), quota_log

    return pd.concat(all_frames, ignore_index=True), quota_log


# -----------------------------
# Display helpers
# -----------------------------

def clean_market_name(market: str) -> str:
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


def format_american(value: Any) -> str:
    value = safe_float(value)

    if value is None:
        return "—"

    value = int(round(value))

    if value > 0:
        return f"+{value}"

    return str(value)


def build_bet_label(row: pd.Series) -> str:
    market = clean_market_name(row.get("market", row.get("market_key", "")))
    selection = row.get("selection", "")
    description = row.get("description", "")
    point = row.get("point", None)

    pieces = []

    if description and not pd.isna(description):
        pieces.append(str(description))

    if selection and not pd.isna(selection):
        pieces.append(str(selection))

    if point is not None and not pd.isna(point):
        pieces.append(str(point))

    label = " ".join(pieces).strip()

    if not label:
        label = str(selection)

    return f"{market}: {label}"


def format_opportunities_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    out["commence_time"] = pd.to_datetime(
        out["commence_time"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert("America/Chicago")

    out["Start Time"] = out["commence_time"].dt.strftime("%a %b %d, %I:%M %p CT")
    out["Bet"] = out.apply(build_bet_label, axis=1)
    out["Odds"] = out["price"].apply(format_american)
    out["Market Fair Odds"] = out["fair_american_odds"].apply(format_american)
    out["Predictive Fair Odds"] = out["predictive_fair_american_odds"].apply(format_american)
    out["Blended Fair Odds"] = out["blended_fair_american_odds"].apply(format_american)
    out["Alpha Fair Odds"] = out["alpha_fair_american_odds"].apply(format_american)

    out["Market EV %"] = out["market_ev_percent"].apply(
        lambda x: "—" if pd.isna(x) else f"{x:.2f}%"
    )

    out["Predictive EV %"] = out["predictive_ev_percent"].apply(
        lambda x: "—" if pd.isna(x) else f"{x:.2f}%"
    )

    out["Blended EV %"] = out["blended_ev_percent"].apply(
        lambda x: "—" if pd.isna(x) else f"{x:.2f}%"
    )

    out["Alpha EV %"] = out["alpha_ev_percent"].apply(
        lambda x: "—" if pd.isna(x) else f"{x:.2f}%"
    )

    out["Suggested Bet"] = out["suggested_bet_size"].apply(
        lambda x: f"${x:.2f}"
    )

    out["Market Edge"] = out["market_edge_probability_points"].apply(
        lambda x: "—" if pd.isna(x) else f"{x:.2f} pts"
    )

    out["Market Fair Prob"] = out["fair_probability"].apply(
        lambda x: "—" if pd.isna(x) else f"{x:.2%}"
    )

    out["Predictive Prob"] = out["predictive_probability"].apply(
        lambda x: "—" if pd.isna(x) else f"{x:.2%}"
    )

    out["Blended Prob"] = out["blended_probability"].apply(
        lambda x: "—" if pd.isna(x) else f"{x:.2%}"
    )

    out["Quality Score"] = out["bet_quality_score"].apply(
        lambda x: "—" if pd.isna(x) else f"{x:.1f}/100"
    )

    cols = [
        "bet_tier",
        "Quality Score",
        "Alpha EV %",
        "Blended EV %",
        "Predictive EV %",
        "Market EV %",
        "Suggested Bet",
        "book",
        "Bet",
        "Odds",
        "Alpha Fair Odds",
        "Blended Fair Odds",
        "Market Fair Odds",
        "Predictive Fair Odds",
        "Market Edge",
        "Market Fair Prob",
        "Predictive Prob",
        "Blended Prob",
        "warning_flags",
        "sport",
        "Start Time",
        "game",
        "market",
        "reference_book_count",
        "model_type",
        "model_away_lambda",
        "model_home_lambda",
        "model_total_lambda",
        "model_total_line",
    ]

    return out[[c for c in cols if c in out.columns]]