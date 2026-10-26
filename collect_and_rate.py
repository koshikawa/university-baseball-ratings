#!/usr/bin/env python3
"""Collect official Japanese university baseball results and fit time-weighted BT ratings.

Scope: 2016 spring through 2026 spring. Sources are the official league/tournament
sites. The script writes normalized games, coverage logs, ratings, histories, and
model validation CSV files. It is intentionally self-contained for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs

import numpy as np
import pandas as pd
import requests
from lxml import html
from scipy.optimize import minimize


START_YEAR = 2016
END_YEAR = 2026
AS_OF = pd.Timestamp("2026-06-30")
RATING_SCALE = 400.0 / math.log(10.0)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; academic rating research/1.0)"})
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


ALIASES = {
    # Tokyo Big6
    "東大": "東京大学", "東京大": "東京大学", "早大": "早稲田大学", "早稲田大": "早稲田大学",
    "慶大": "慶應義塾大学", "慶應大": "慶應義塾大学", "慶應義塾大": "慶應義塾大学",
    "明大": "明治大学", "明治大": "明治大学", "法大": "法政大学", "法政大": "法政大学",
    "立大": "立教大学", "立教大": "立教大学",
    # Tohto
    "青学大": "青山学院大学", "青山学院大": "青山学院大学", "亜大": "亜細亜大学", "亜細亜大": "亜細亜大学",
    "國學院": "國學院大學", "國學院大": "國學院大學", "國學院大學": "國學院大學",
    "中大": "中央大学", "中央大": "中央大学", "日大": "日本大学", "日本大": "日本大学",
    "東洋大": "東洋大学", "駒大": "駒澤大学", "駒澤大": "駒澤大学", "専大": "専修大学", "専修大": "専修大学",
    "国士大": "国士舘大学", "国士舘大": "国士舘大学", "拓大": "拓殖大学", "拓殖大": "拓殖大学",
    "東農大": "東京農業大学", "東京農大": "東京農業大学", "立正大": "立正大学", "大正大": "大正大学",
    "帝平大": "帝京平成大学", "帝京平成大": "帝京平成大学", "順大": "順天堂大学", "順天堂大": "順天堂大学",
    "学習大": "学習院大学", "学習院大": "学習院大学", "成蹊大": "成蹊大学", "芝工大": "芝浦工業大学",
    "芝浦工大": "芝浦工業大学", "上智大": "上智大学", "一橋大": "一橋大学", "東工大": "東京科学大学",
    "東京工大": "東京科学大学", "東京工業大": "東京科学大学", "東京科学大": "東京科学大学", "科学大": "東京科学大学",
    "都市大": "東京都市大学", "東京都市大": "東京都市大学",
    # Shuto
    "日体大": "日本体育大学", "日本体育大": "日本体育大学", "筑波大": "筑波大学", "東海大": "東海大学",
    "桜美大": "桜美林大学", "桜美林大": "桜美林大学", "城西大": "城西大学", "帝京大": "帝京大学",
    "武蔵大": "武蔵大学", "明星大": "明星大学", "東経大": "東京経済大学", "東京経済大": "東京経済大学",
    "大東大": "大東文化大学", "大東文化大": "大東文化大学", "獨協大": "獨協大学", "成城大": "成城大学",
    "玉川大": "玉川大学", "足工大": "足利大学", "足利工大": "足利大学", "足利大": "足利大学",
    "明学大": "明治学院大学", "明治学院大": "明治学院大学", "文教大": "文教大学",
    "ウェ大": "日本ウェルネススポーツ大学", "ウェルネス大": "日本ウェルネススポーツ大学", "日本ウェルネス大": "日本ウェルネススポーツ大学",
    "日本ウェルネススポーツ大": "日本ウェルネススポーツ大学",
    # Frequent national-tournament aliases
    "東北福祉大": "東北福祉大学", "上武大": "上武大学", "創価大": "創価大学", "神奈川大": "神奈川大学",
    "桐蔭横浜大": "桐蔭横浜大学", "中央学院大": "中央学院大学", "国際武道大": "国際武道大学",
    "東日本国際大": "東日本国際大学", "富士大": "富士大学", "八戸学院大": "八戸学院大学",
    "仙台大": "仙台大学", "東海大北海道": "東海大学北海道キャンパス", "東農大北海道": "東京農業大学北海道オホーツク",
    "福井工業大": "福井工業大学", "中部学院大": "中部学院大学", "中京学院大": "中京学院大学",
    "中京大": "中京大学", "名城大": "名城大学", "愛知学院大": "愛知学院大学", "愛知工業大": "愛知工業大学",
    "皇學館大": "皇學館大学", "京都産業大": "京都産業大学", "立命館大": "立命館大学", "関西大": "関西大学",
    "関西国際大": "関西国際大学", "大阪商業大": "大阪商業大学", "大阪体育大": "大阪体育大学",
    "天理大": "天理大学", "佛教大": "佛教大学", "環太平洋大": "環太平洋大学", "広島経済大": "広島経済大学",
    "九州産業大": "九州産業大学", "九州共立大": "九州共立大学", "九州国際大": "九州国際大学",
    "日本文理大": "日本文理大学", "奈良学園大": "奈良学園大学", "共栄大": "共栄大学",
    "第一工業大": "第一工科大学", "第一工科大": "第一工科大学", "愛媛大": "愛媛大学",
}


def clean_text(x: str) -> str:
    x = re.sub(r"[\u3000\xa0\s]+", " ", str(x)).strip()
    return x


def canonical_team(x: str) -> str:
    x = clean_text(x)
    x = re.sub(r"^[※＊*○●△\s]+", "", x)
    x = re.sub(r"[○●△※＊*\s]+$", "", x)
    x = x.replace("（", "(").replace("）", ")")
    x = re.sub(r"\s*\([^)]*\)$", "", x).strip()
    if x in ALIASES:
        return ALIASES[x]
    # National pages commonly abbreviate 大学 as 大.
    if x.endswith("大") and not x.endswith("大大"):
        return x[:-1] + "大学"
    return x


def make_date(year: int, md: str) -> pd.Timestamp | None:
    m = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})", clean_text(md))
    if not m:
        return None
    try:
        return pd.Timestamp(year=year, month=int(m.group(1)), day=int(m.group(2)))
    except ValueError:
        return None


def get(url: str, *, encoding: str | None = None, verify: bool = True, tries: int = 3) -> requests.Response:
    last = None
    for n in range(tries):
        try:
            r = SESSION.get(url, timeout=90, verify=verify)
            r.raise_for_status()
            if encoding:
                r.encoding = encoding
            return r
        except Exception as e:
            last = e
            time.sleep(1.2 * (n + 1))
    raise RuntimeError(f"GET failed: {url}: {last}")


def game_row(**kw):
    base = dict(
        date=None, season="", competition="", league="", division="", phase="",
        team1="", score1=np.nan, team2="", score2=np.nan, result="", status="played",
        source_url="", source_type="official", notes=""
    )
    base.update(kw)
    return base


def parse_big6(year: int, term: str):
    code = f"{year}{'s' if term == '春' else 'a'}"
    url = f"https://www.big6.gr.jp/game/league/{code}/{code}_schedule.html"
    out, log = [], {"source": "東京六大学野球連盟", "url": url, "status": "ok", "rows": 0, "note": ""}
    try:
        text = get(url, encoding="utf-8", verify=False).text
        root = html.fromstring(text)
        for tr in root.xpath("//tr[td[contains(@class,'scd_date')]]"):
            date_text = clean_text(" ".join(tr.xpath("./td[contains(@class,'scd_date')]//text()")))
            dt = make_date(year, date_text)
            cells = tr.xpath(".//td")
            bits = []
            for td in cells:
                cls = td.get("class", "")
                if "scd_vs" in cls:
                    normal = td.xpath(".//span[contains(@class,'display_none_on_sp')]/text()")
                    bits.append(clean_text(normal[0] if normal else " ".join(td.xpath(".//text()"))))
            for i in range(0, len(bits) - 2, 3):
                t1, sc, t2 = bits[i:i+3]
                sm = re.fullmatch(r"(\d+)\s*-\s*(\d+)", sc)
                if sm:
                    s1, s2 = map(int, sm.groups())
                    res = "draw" if s1 == s2 else ("team1" if s1 > s2 else "team2")
                    out.append(game_row(date=dt, season=f"{year}{term}", competition="リーグ戦", league="東京六大学", division="1部",
                                        phase="league", team1=canonical_team(t1), score1=s1, team2=canonical_team(t2), score2=s2,
                                        result=res, source_url=url))
                elif t1 and t2 and ("中止" in sc or sc in {"-", "－"}):
                    out.append(game_row(date=dt, season=f"{year}{term}", competition="リーグ戦", league="東京六大学", division="1部",
                                        phase="league", team1=canonical_team(t1), team2=canonical_team(t2), status="canceled",
                                        result="not_played", source_url=url, notes=sc))
        log["rows"] = len(out)
    except Exception as e:
        log.update(status="error", note=str(e))
    return out, log


def extract_dated_match_rows(text: str, year: int, url: str, league: str, division: str, season: str, competition: str, phase: str):
    root = html.fromstring(text)
    out = []
    seen = set()
    for tr in root.xpath("//tr"):
        cells = [clean_text(" ".join(td.xpath(".//text()"))) for td in tr.xpath("./th|./td")]
        if not cells:
            continue
        date_cell = next((c for c in cells if re.search(r"\d{1,2}\s*/\s*\d{1,2}", c)), None)
        if not date_cell:
            continue
        dt = make_date(year, date_cell)
        # Prefer the shortest cell containing a numeric score and two names.
        candidates = sorted([c for c in cells if re.search(r"\d+\s*[-－]\s*\d+", c)], key=len)
        found = False
        for c in candidates:
            z = clean_text(c.replace("詳細", "").replace("※", ""))
            m = re.search(r"(.+?)\s+(\d+)\s*[-－]\s*(\d+)\s+(.+)", z)
            if not m:
                continue
            t1, s1, s2, t2 = m.groups()
            t1, t2 = canonical_team(t1), canonical_team(t2)
            if not t1 or not t2 or len(t1) > 30 or len(t2) > 30:
                continue
            s1, s2 = int(s1), int(s2)
            key = (dt, t1, s1, t2, s2, competition, division)
            if key in seen:
                found = True
                break
            seen.add(key)
            out.append(game_row(date=dt, season=season, competition=competition, league=league, division=division, phase=phase,
                                team1=t1, score1=s1, team2=t2, score2=s2,
                                result="draw" if s1 == s2 else ("team1" if s1 > s2 else "team2"), source_url=url))
            found = True
            break
        if found:
            continue
        # Record scheduled cancellations where identifiable.
        cc = sorted([c for c in cells if "中止" in c and re.search(r"[-－]", c)], key=len)
        if cc:
            z = clean_text(cc[0].replace("中止", "").replace("詳細", ""))
            m = re.search(r"(.+?)\s+[-－]\s+(.+)", z)
            if m:
                t1, t2 = map(canonical_team, m.groups())
                key = (dt, t1, t2, "canceled", competition, division)
                if key not in seen:
                    seen.add(key)
                    out.append(game_row(date=dt, season=season, competition=competition, league=league, division=division, phase=phase,
                                        team1=t1, team2=t2, result="not_played", status="canceled", source_url=url, notes="中止"))
    return out


def parse_tohto(year: int, term: str, div: int):
    sid = "01" if term == "春" else "02"
    url = f"https://www.tohto-bbl.com/gameinfo/schedule.php?YEAR={year}&SEASONID={sid}&LEAGUEID={div:02d}"
    log = {"source": "東都大学野球連盟", "url": url, "status": "ok", "rows": 0, "note": ""}
    try:
        text = get(url, encoding="shift_jis", verify=False).text
        out = extract_dated_match_rows(text, year, url, "東都", f"{div}部", f"{year}{term}", "リーグ戦", "league")
        log["rows"] = len(out)
    except Exception as e:
        out = []
        if "No tables found" in str(e):
            log.update(status="no_table", note="公式ページに日程表なし（中止または分割ページで代替）")
        else:
            log.update(status="error", note=str(e))
    return out, log


def parse_tohto_playoff(year: int, term: str):
    suffix = "s" if term == "春" else "a"
    url = f"https://www.tohto-bbl.com/gameinfo/{year}/change_{suffix}.shtml"
    log = {"source": "東都大学野球連盟", "url": url, "status": "ok", "rows": 0, "note": ""}
    try:
        text = get(url, encoding="shift_jis", verify=False).text
        out = extract_dated_match_rows(text, year, url, "東都", "入替戦", f"{year}{term}", "入替戦", "promotion")
        log["rows"] = len(out)
    except Exception as e:
        out = []
        log.update(status="error", note=str(e))
    return out, log


def parse_shuto_page(year: int, term: str, ds: str, division: str, competition: str, phase: str):
    stem = f"{year}{'haru' if term == '春' else 'aki'}"
    path = f"{year}{term}" if year >= 2024 else stem
    url = f"https://tmubl.jp/game/{path}?ds={ds}"
    log = {"source": "首都大学野球連盟", "url": url, "status": "ok", "rows": 0, "note": ""}
    try:
        response = get(url)
        text = response.text
        tables = pd.read_html(BytesIO(response.content), flavor="lxml")
        out = []
        for t in tables:
            cols = [clean_text(c) for c in t.columns]
            if "対戦カード" not in cols:
                continue
            date_col = next((c for c in t.columns if clean_text(c).startswith("Unnamed") or clean_text(c) in {"月日", "日程"}), t.columns[0])
            match_col = t.columns[cols.index("対戦カード")]
            for _, row in t.iterrows():
                dtext, match = clean_text(row.get(date_col, "")), clean_text(row.get(match_col, ""))
                dt = make_date(year, dtext)
                if dt is None or not match or match == "nan":
                    continue
                m = re.search(r"(.+?)\s+(\d+)\s*[-－]\s*(\d+)\s+(.+?)(?:\s+詳細)?$", match)
                if m:
                    t1, s1, s2, t2 = m.groups(); s1, s2 = int(s1), int(s2)
                    out.append(game_row(date=dt, season=f"{year}{term}", competition=competition, league="首都", division=division,
                                        phase=phase, team1=canonical_team(t1), score1=s1, team2=canonical_team(t2), score2=s2,
                                        result="draw" if s1 == s2 else ("team1" if s1 > s2 else "team2"), source_url=url))
                elif "中止" in match:
                    z = match.replace("雨天中止", "-").replace("中止", "-")
                    m = re.search(r"(.+?)\s+[-－]\s+(.+)", z)
                    if m:
                        out.append(game_row(date=dt, season=f"{year}{term}", competition=competition, league="首都", division=division,
                                            phase=phase, team1=canonical_team(m.group(1)), team2=canonical_team(m.group(2)),
                                            result="not_played", status="canceled", source_url=url, notes="中止"))
        out = pd.DataFrame(out).drop_duplicates(subset=["date","team1","score1","team2","score2","competition"]).to_dict("records") if out else []
        log["rows"] = len(out)
    except Exception as e:
        out = []
        if "No tables found" in str(e):
            log.update(status="no_table", note="公式ページに日程表なし（中止または分割ページで代替）")
        else:
            log.update(status="error", note=str(e))
    return out, log


def shuto_playoff_ds(year: int, term: str):
    stem = f"{year}{'haru' if term == '春' else 'aki'}"
    url = f"https://tmubl.jp/game/{stem}?ds={stem}-1"
    try:
        root = html.fromstring(get(url).content)
        hrefs = root.xpath("//a[contains(@href,'irekae')]/@href")
        if hrefs:
            return parse_qs(urlparse(hrefs[0]).query).get("ds", [hrefs[0].split("ds=")[-1]])[0]
    except Exception:
        pass
    return None


def discover_shuto_pages(year: int, term: str):
    """Return official season page identifiers, including split-group seasons."""
    stem = f"{year}{'haru' if term == '春' else 'aki'}"
    path = f"{year}{term}" if year >= 2024 else stem
    seeds = [f"{stem}-1", f"{stem}-2"]
    if year == 2016 and term == "秋":
        seeds[0] = "2016ai-1"  # historical official-site typo retained in its URL
    found = set(seeds)
    for ds in list(seeds):
        try:
            root = html.fromstring(get(f"https://tmubl.jp/game/{path}?ds={ds}").content)
            for href in root.xpath("//a[contains(@href,'ds=')]/@href"):
                val = parse_qs(urlparse(href).query).get("ds", [""])[0]
                if val.startswith(str(year)) and not any(bad in val for bad in ["sinzin", "shinjin"]):
                    found.add(val)
        except Exception:
            pass
    # Known split-group pages were not always linked consistently in archived navigation.
    if year == 2019:
        found.update({f"{stem}-2w", f"{stem}-2r", f"{stem}-kettei" if term == "春" else f"{stem}-ketteisen"})
    if year == 2020 and term == "秋":
        found.update({"2020aki-2w", "2020aki-2r", "2020aki-ketteisen"})
    if year == 2021 and term == "秋":
        found.update({"2021aki-2w", "2021aki-2r", "2021aki-ketteisen"})
    pages=[]
    for ds in sorted(found):
        if "irekae" in ds or "irekjae" in ds:
            pages.append((ds,"入替戦","入替戦","promotion"))
        elif ds.endswith("-1") or ds == "2016ai-1":
            pages.append((ds,"1部","リーグ戦","league"))
        elif any(x in ds for x in ["-2", "kettei"]):
            pages.append((ds,"2部","リーグ戦","league"))
    return pages


def parse_national(year: int, kind: str):
    if kind == "alljapan":
        url = f"https://www.jubf.net/system/prog/schedule.php?k=all&m=pc&s={year}"
        source, competition, month = "全日本大学野球連盟", "全日本大学野球選手権", 6
    else:
        url = f"https://www.student-baseball.or.jp/system/prog/schedule.php?e=jingu&k=all&m=pc&s={year}"
        source, competition, month = "日本学生野球協会", "明治神宮野球大会", 11
    log = {"source": source, "url": url, "status": "ok", "rows": 0, "note": ""}
    out = []
    try:
        text = get(url).text
        root = html.fromstring(text)
        links = root.xpath("//a[contains(@href,'game.php')]")
        seen = set()
        for a in links:
            href = a.get("href", "")
            q = parse_qs(urlparse(href).query)
            gd = q.get("gd", [None])[0]
            if not gd:
                continue
            if kind == "jingu":
                try:
                    gid = int(q.get("gid", [0])[0])
                except ValueError:
                    gid = 0
                if gid < 20:  # high-school games use the 10-series.
                    continue
                # Score/team container is the nearest ancestor whose direct children contain three bold spans.
                node = a
                container = None
                for _ in range(6):
                    node = node.getparent()
                    if node is None: break
                    spans = node.xpath("./div/span[contains(@class,'text14px')] | .//span[contains(@class,'text14px')]")
                    vals = [clean_text(" ".join(s.xpath(".//text()"))) for s in spans]
                    if any(re.fullmatch(r"\d+\s*-\s*\d+", v) for v in vals) and len(vals) >= 3:
                        container = node; break
                if container is None:
                    continue
                vals = [clean_text(" ".join(s.xpath(".//text()"))) for s in container.xpath(".//span[contains(@class,'text14px')]")]
                si = next((i for i,v in enumerate(vals) if re.fullmatch(r"\d+\s*-\s*\d+", v)), None)
                if si is None or si == 0 or si + 1 >= len(vals):
                    continue
                t1, sc, t2 = vals[si-1], vals[si], vals[si+1]
            else:
                tr = a
                while tr is not None and tr.tag.lower() != "tr":
                    tr = tr.getparent()
                if tr is None:
                    continue
                cells = [clean_text(" ".join(td.xpath(".//text()"))) for td in tr.xpath("./td")]
                si = next((i for i,v in enumerate(cells) if re.fullmatch(r"\d+\s*-\s*\d+", v)), None)
                if si is None or si == 0 or si+1 >= len(cells):
                    continue
                t1, sc, t2 = cells[si-1], cells[si], cells[si+1]
            sm = re.fullmatch(r"(\d+)\s*-\s*(\d+)", sc)
            if not sm:
                continue
            dt = pd.Timestamp(gd)
            s1, s2 = map(int, sm.groups())
            t1, t2 = canonical_team(t1), canonical_team(t2)
            key = (dt, t1, s1, t2, s2, competition)
            if key in seen:
                continue
            seen.add(key)
            out.append(game_row(date=dt, season=f"{year}{'春' if month == 6 else '秋'}", competition=competition,
                                league="全国", division="全国大会", phase="national", team1=t1, score1=s1, team2=t2, score2=s2,
                                result="draw" if s1 == s2 else ("team1" if s1 > s2 else "team2"), source_url=url))
        log["rows"] = len(out)
        if len(out) == 0 and year == 2020:
            log.update(status="canceled", note="新型コロナウイルスの影響で大会中止")
    except Exception as e:
        log.update(status="error", note=str(e))
    return out, log


def collect():
    games, logs = [], []
    seasons = [(y, t) for y in range(START_YEAR, END_YEAR+1) for t in ("春", "秋") if not (y == END_YEAR and t == "秋")]
    for y, term in seasons:
        rows, lg = parse_big6(y, term); games += rows; logs.append(lg)
        for div in range(1, 5):
            rows, lg = parse_tohto(y, term, div); games += rows; logs.append(lg)
        rows, lg = parse_tohto_playoff(y, term); games += rows; logs.append(lg)
        for ds,division,competition,phase in discover_shuto_pages(y,term):
            rows, lg = parse_shuto_page(y,term,ds,division,competition,phase)
            games += rows; logs.append(lg)
    for y in range(START_YEAR, END_YEAR+1):
        rows, lg = parse_national(y, "alljapan"); games += rows; logs.append(lg)
    for y in range(START_YEAR, END_YEAR):
        rows, lg = parse_national(y, "jingu"); games += rows; logs.append(lg)
    df = pd.DataFrame(games)
    if df.empty:
        raise RuntimeError("No games collected")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Remove exact duplicates caused by responsive/mobile duplicate markup.
    subset = ["date","competition","league","division","team1","score1","team2","score2","status"]
    df = df.drop_duplicates(subset=subset).sort_values(["date","competition","team1","team2"]).reset_index(drop=True)
    df.insert(0, "game_id", [f"G{i+1:05d}" for i in range(len(df))])
    logdf = pd.DataFrame(logs)
    return df, logdf


def season_key(ts: pd.Timestamp):
    return f"{ts.year}{'春' if ts.month <= 7 else '秋'}"


def prepare_played(games: pd.DataFrame):
    g = games[(games.status == "played") & games.score1.notna() & games.score2.notna()].copy()
    g = g[g.date <= AS_OF].copy()
    g["y"] = np.where(g.score1 > g.score2, 1.0, np.where(g.score1 < g.score2, 0.0, 0.5))
    return g


def fit_bt(g: pd.DataFrame, as_of: pd.Timestamp, half_life_years: float, ridge_games: float, teams=None, return_cov=False):
    g = g[g.date <= as_of].copy()
    if teams is None:
        teams = sorted(set(g.team1) | set(g.team2))
    idx = {t:i for i,t in enumerate(teams)}
    g = g[g.team1.isin(idx) & g.team2.isin(idx)].copy()
    i = g.team1.map(idx).to_numpy(); j = g.team2.map(idx).to_numpy(); y = g.y.to_numpy(float)
    age_years = np.maximum(0, (as_of - g.date).dt.days.to_numpy() / 365.25)
    w = np.exp(-math.log(2) * age_years / half_life_years)
    n = len(teams)
    def fg(x):
        d = x[i] - x[j]
        p = 1/(1+np.exp(-np.clip(d, -30, 30)))
        loss = -np.sum(w * (y*np.log(np.clip(p,1e-12,1)) + (1-y)*np.log(np.clip(1-p,1e-12,1)))) + 0.5*ridge_games*np.dot(x,x)
        z = w*(p-y)
        grad = ridge_games*x.copy()
        np.add.at(grad, i, z); np.add.at(grad, j, -z)
        return loss, grad
    res = minimize(lambda x: fg(x)[0], np.zeros(n), jac=lambda x: fg(x)[1], method="L-BFGS-B", options={"maxiter":1000})
    x = res.x - np.mean(res.x)
    rating = 1500 + RATING_SCALE*x
    covr = None
    if return_cov:
        d=x[i]-x[j]; p=1/(1+np.exp(-np.clip(d,-30,30))); h=w*p*(1-p)
        H=np.eye(n)*ridge_games
        np.add.at(H,(i,i),h); np.add.at(H,(j,j),h); np.add.at(H,(i,j),-h); np.add.at(H,(j,i),-h)
        covx=np.linalg.pinv(H)
        covr=covx*(RATING_SCALE**2)
    return pd.Series(rating,index=teams), covr, {"success":bool(res.success),"objective":float(res.fun),"games":len(g),"weight_sum":float(w.sum())}


def validate_model(g: pd.DataFrame):
    candidates=[]
    cutoffs=[pd.Timestamp("2022-04-01"),pd.Timestamp("2023-04-01"),pd.Timestamp("2024-04-01"),pd.Timestamp("2025-04-01")]
    for hl in [1.0,1.5,2.0,3.0,4.0]:
        for ridge in [0.5,1.0,2.0,5.0,10.0]:
            preds=[]
            for c in cutoffs:
                train=g[g.date<c]; test=g[(g.date>=c)&(g.date<c+pd.DateOffset(years=1))]
                if train.empty or test.empty: continue
                teams=sorted(set(train.team1)|set(train.team2))
                r,_,_=fit_bt(train,c,hl,ridge,teams=teams)
                known=test.team1.isin(r.index)&test.team2.isin(r.index)
                tt=test[known]
                if tt.empty: continue
                d=(tt.team1.map(r)-tt.team2.map(r)).to_numpy()/RATING_SCALE
                p=1/(1+np.exp(-np.clip(d,-30,30))); y=tt.y.to_numpy()
                for pp,yy in zip(p,y): preds.append((pp,yy))
            arr=np.array(preds)
            if len(arr):
                p=np.clip(arr[:,0],1e-9,1-1e-9); y=arr[:,1]
                candidates.append({"half_life_years":hl,"ridge_games":ridge,"n_predictions":len(arr),
                                   "log_loss":float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))),
                                   "brier":float(np.mean((p-y)**2)),
                                   "accuracy_ex_draws":float(np.mean((p[y!=.5]>=.5)==(y[y!=.5]==1))) if np.any(y!=.5) else np.nan})
    return pd.DataFrame(candidates).sort_values(["log_loss","brier"]).reset_index(drop=True)


def current_membership(games: pd.DataFrame):
    x=games[(games.season=="2026春")&(games.competition=="リーグ戦")&(games.status=="played")]
    rows=[]
    for (league,division),z in x.groupby(["league","division"]):
        for t in sorted(set(z.team1)|set(z.team2)):
            rows.append({"team":t,"league_2026_spring":league,"division_2026_spring":division})
    return pd.DataFrame(rows).drop_duplicates("team")


def rate_all(games: pd.DataFrame):
    g=prepare_played(games)
    val=validate_model(g)
    best=val.iloc[0]
    hl=float(best.half_life_years); ridge=float(best.ridge_games)
    ratings,cov,diag=fit_bt(g,AS_OF,hl,ridge,return_cov=True)
    se=np.sqrt(np.maximum(0,np.diag(cov)))
    info=pd.DataFrame({"team":ratings.index,"rating":ratings.values,"se":se,
                       "ci95_low":ratings.values-1.96*se,"ci95_high":ratings.values+1.96*se})
    n_games=pd.concat([g.team1,g.team2]).value_counts()
    recent=g[g.date>=AS_OF-pd.DateOffset(years=2)]
    n_recent=pd.concat([recent.team1,recent.team2]).value_counts()
    info["games_2016_2026"]=info.team.map(n_games).fillna(0).astype(int)
    info["games_last_2y"]=info.team.map(n_recent).fillna(0).astype(int)
    memb=current_membership(games)
    info=info.merge(memb,on="team",how="left")
    target=info[info.league_2026_spring.isin(["東都","東京六大学","首都"])].copy()
    target["rank_all_target"]=target.rating.rank(ascending=False,method="min").astype(int)
    target=target.sort_values("rating",ascending=False)
    # Historical snapshots after every completed spring/fall season.
    snapshots=[]
    for y in range(START_YEAR,END_YEAR+1):
        for term,md in [("春",f"{y}-06-30"),("秋",f"{y}-12-15")]:
            if y==END_YEAR and term=="秋": continue
            cutoff=pd.Timestamp(md)
            gg=g[g.date<=cutoff]
            if gg.empty: continue
            rr,_,_=fit_bt(gg,cutoff,hl,ridge)
            for t,v in rr.items(): snapshots.append({"season":f"{y}{term}","as_of":cutoff,"team":t,"rating":v})
    hist=pd.DataFrame(snapshots)
    # Tokyo-to-Tohto equivalence with covariance-based Monte Carlo.
    eq=[]
    if "東京大学" in ratings.index:
        rng=np.random.default_rng(20261009)
        sample=rng.multivariate_normal(ratings.values,cov,size=10000,check_valid="ignore")
        ti=list(ratings.index).index("東京大学")
        tohto=target[target.league_2026_spring=="東都"].copy()
        divs=["1部","2部","3部","4部"]
        div_indices={d:[list(ratings.index).index(t) for t in tohto[tohto.division_2026_spring==d].team if t in ratings.index] for d in divs}
        winners=[]
        for s in sample:
            meds={d:np.median(s[ii]) for d,ii in div_indices.items() if ii}
            winners.append(min(meds,key=lambda d:abs(s[ti]-meds[d])))
        for d in divs:
            members=tohto[tohto.division_2026_spring==d]
            eq.append({"division":d,"median_rating":members.rating.median(),"min_rating":members.rating.min(),"max_rating":members.rating.max(),
                       "probability_nearest_division":winners.count(d)/len(winners)})
    eq=pd.DataFrame(eq)
    return target, info.sort_values("rating",ascending=False), hist, val, eq, {"half_life_years":hl,"ridge_games":ridge,**diag}


def audit(games: pd.DataFrame, logs: pd.DataFrame):
    played=games[games.status=="played"]
    issues=[]
    if played.date.isna().any(): issues.append({"issue":"played_missing_date","count":int(played.date.isna().sum())})
    if (played.team1==played.team2).any(): issues.append({"issue":"same_team","count":int((played.team1==played.team2).sum())})
    if played[["score1","score2"]].isna().any(axis=1).any(): issues.append({"issue":"played_missing_score","count":int(played[["score1","score2"]].isna().any(axis=1).sum())})
    dup=played.duplicated(["date","team1","score1","team2","score2","competition"]).sum()
    issues.append({"issue":"exact_duplicate_games","count":int(dup)})
    issues.append({"issue":"source_pages_error","count":int((logs.status=="error").sum())})
    issues.append({"issue":"source_pages_missing_or_canceled","count":int(logs.status.isin(["missing","canceled"]).sum())})
    return pd.DataFrame(issues)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="work/data"); ap.add_argument("--reuse",action="store_true")
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if args.reuse and (out/"games.csv").exists():
        games=pd.read_csv(out/"games.csv",parse_dates=["date"]); logs=pd.read_csv(out/"source_log.csv")
    else:
        games,logs=collect(); games.to_csv(out/"games.csv",index=False); logs.to_csv(out/"source_log.csv",index=False)
    aud=audit(games,logs); aud.to_csv(out/"audit.csv",index=False)
    target,allr,hist,val,eq,params=rate_all(games)
    target.to_csv(out/"ratings_target_latest.csv",index=False)
    allr.to_csv(out/"ratings_all_latest.csv",index=False)
    hist.to_csv(out/"rating_history.csv",index=False)
    val.to_csv(out/"model_validation.csv",index=False)
    eq.to_csv(out/"tokyo_tohto_equivalence.csv",index=False)
    (out/"model_parameters.json").write_text(json.dumps(params,ensure_ascii=False,indent=2),encoding="utf-8")
    for name, frame in {
        "games": games, "source_log": logs, "audit": aud,
        "ratings_target_latest": target, "ratings_all_latest": allr,
        "rating_history": hist, "model_validation": val,
        "tokyo_tohto_equivalence": eq,
    }.items():
        payload = json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))
        (out/f"{name}.json").write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"games":len(games),"played":int((games.status=='played').sum()),"canceled":int((games.status=='canceled').sum()),
                      "source_pages":len(logs),"errors":int((logs.status=='error').sum()),"params":params},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
