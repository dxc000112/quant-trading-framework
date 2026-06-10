"""
GEX Scanner — 基于 Charles Schwab API 的实时 Gamma 结构分析
════════════════════════════════════════════════════════════

第一次使用前的准备（只需做一次）：
  1. 去 https://developer.schwab.com 注册，创建 App
  2. 把 App Key 和 App Secret 填到 .env 里：
       SCHWAB_APP_KEY=你的AppKey
       SCHWAB_APP_SECRET=你的AppSecret
  3. 第一次运行时会打开浏览器让你授权，之后自动刷新 token

用法:
  python gex_scanner.py          # 默认分析 NVDA
  python gex_scanner.py SPY      # 分析 SPY
"""

import os
import sys
import json
import time
import base64
import warnings
import webbrowser
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse, parse_qs

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.stats import norm
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# ── 常量 ────────────────────────────────────────────────────────────
SCHWAB_BASE      = 'https://api.schwabapi.com'
SCHWAB_AUTH_URL  = 'https://api.schwabapi.com/v1/oauth/authorize'
SCHWAB_TOKEN_URL = 'https://api.schwabapi.com/v1/oauth/token'
SCHWAB_REDIRECT  = 'https://127.0.0.1'          # 和 developer portal 里填的一致
TOKEN_FILE       = os.path.join(os.path.dirname(__file__), '.schwab_token.json')

RISK_FREE_RATE   = 0.05
CONTRACT_SIZE    = 100

APP_KEY    = os.getenv('SCHWAB_APP_KEY', '')
APP_SECRET = os.getenv('SCHWAB_APP_SECRET', '')


# ════════════════════════════════════════════════════════════════════
#  OAuth2 认证层
# ════════════════════════════════════════════════════════════════════

def _b64_credentials() -> str:
    return base64.b64encode(f'{APP_KEY}:{APP_SECRET}'.encode()).decode()


def _save_token(token: dict):
    token['saved_at'] = time.time()
    with open(TOKEN_FILE, 'w') as f:
        json.dump(token, f, indent=2)


def _load_token() -> dict | None:
    if not os.path.exists(TOKEN_FILE):
        return None
    with open(TOKEN_FILE) as f:
        return json.load(f)


def _refresh_token(token: dict) -> dict:
    """用 refresh_token 换新的 access_token（无需浏览器）。"""
    resp = requests.post(
        SCHWAB_TOKEN_URL,
        headers={
            'Authorization': f'Basic {_b64_credentials()}',
            'Content-Type':  'application/x-www-form-urlencoded',
        },
        data={
            'grant_type':    'refresh_token',
            'refresh_token': token['refresh_token'],
        },
        timeout=15,
    )
    resp.raise_for_status()
    new_token = resp.json()
    _save_token(new_token)
    print('  ✓ Token 自动刷新成功')
    return new_token


def _first_time_login() -> dict:
    """第一次授权：打开浏览器 → 用户点允许 → 粘贴回调 URL → 拿 token。"""
    if not APP_KEY or not APP_SECRET:
        raise RuntimeError(
            '请先在 .env 里填写:\n'
            '  SCHWAB_APP_KEY=你的AppKey\n'
            '  SCHWAB_APP_SECRET=你的AppSecret'
        )

    params = {
        'response_type': 'code',
        'client_id':     APP_KEY,
        'redirect_uri':  SCHWAB_REDIRECT,
        'scope':         'readonly',
    }
    auth_url = SCHWAB_AUTH_URL + '?' + urlencode(params)

    print('\n' + '─' * 55)
    print('  首次授权 — 需要完成以下步骤（只需做一次）')
    print('─' * 55)
    print('  1. 浏览器即将打开 Schwab 授权页面')
    print('  2. 登录并点击「允许」')
    print('  3. 浏览器会跳转到一个打不开的页面（127.0.0.1）')
    print('  4. 把那个页面的完整 URL 粘贴到这里')
    print('─' * 55)
    input('  按 Enter 打开浏览器...')
    webbrowser.open(auth_url)

    callback_url = input('\n  粘贴回调 URL: ').strip()
    parsed       = urlparse(callback_url)
    code         = parse_qs(parsed.query).get('code', [''])[0]
    if not code:
        raise RuntimeError('URL 里没有找到 code，请确认粘贴的是完整 URL')

    resp = requests.post(
        SCHWAB_TOKEN_URL,
        headers={
            'Authorization': f'Basic {_b64_credentials()}',
            'Content-Type':  'application/x-www-form-urlencoded',
        },
        data={
            'grant_type':   'authorization_code',
            'code':         code,
            'redirect_uri': SCHWAB_REDIRECT,
        },
        timeout=15,
    )
    resp.raise_for_status()
    token = resp.json()
    _save_token(token)
    print('  ✓ 授权成功，Token 已保存到 .schwab_token.json')
    return token


def get_access_token() -> str:
    """
    自动管理 token 生命周期：
      - 没有 token → 引导首次授权
      - token 过期 → 自动 refresh
      - token 有效 → 直接返回
    """
    token = _load_token()

    if token is None:
        token = _first_time_login()

    # access_token 默认 30 分钟有效，提前 2 分钟刷新
    saved_at    = token.get('saved_at', 0)
    expires_in  = token.get('expires_in', 1800)
    if time.time() - saved_at > expires_in - 120:
        try:
            token = _refresh_token(token)
        except Exception:
            print('  refresh 失败，重新授权...')
            token = _first_time_login()

    return token['access_token']


# ════════════════════════════════════════════════════════════════════
#  Schwab API 请求层
# ════════════════════════════════════════════════════════════════════

_access_token = None


def _schwab_get(path: str, params: dict = None, retries: int = 3) -> dict:
    global _access_token
    if _access_token is None:
        _access_token = get_access_token()

    url = SCHWAB_BASE + path
    for attempt in range(retries):
        try:
            resp = requests.get(
                url,
                headers={'Authorization': f'Bearer {_access_token}'},
                params=params or {},
                timeout=20,
            )
            if resp.status_code == 401:
                # token 失效，强制刷新
                _access_token = get_access_token()
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f'Schwab 请求失败 ({path}): {e}')


# ════════════════════════════════════════════════════════════════════
#  Black-Scholes 工具
# ════════════════════════════════════════════════════════════════════

def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 1e-6 or sigma <= 1e-6 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


# ════════════════════════════════════════════════════════════════════
#  数据拉取
# ════════════════════════════════════════════════════════════════════

def fetch_options(symbol: str, max_expiries: int = 8):
    """
    调用 Schwab /marketdata/v1/chains 接口。
    直接返回 gamma、OI、IV，无需额外计算。
    """
    as_of = pd.Timestamp.now()
    print(f'\n正在拉取 {symbol} 期权链...')

    data = _schwab_get('/marketdata/v1/chains', params={
        'symbol':                  symbol,
        'contractType':            'ALL',
        'includeUnderlyingQuote':  'true',
        'strategy':                'SINGLE',
        'optionType':              'S',          # 标准期权
    })

    # 现价
    underlying = data.get('underlying', {})
    spot = float(underlying.get('mark', 0) or underlying.get('last', 0) or 0)
    if spot <= 0:
        raise RuntimeError(f'无法获取 {symbol} 现价')
    print(f'{symbol}  现价 ${spot:.2f}')

    records = []

    for side, opt_type in [('callExpDateMap', 'C'), ('putExpDateMap', 'P')]:
        exp_map = data.get(side, {})
        # exp_map 结构: { "2026-05-16:15": { "200.0": [{contract}, ...] } }
        for exp_key, strikes in list(exp_map.items())[:max_expiries]:
            expiry_str = exp_key.split(':')[0]
            expiry     = pd.Timestamp(expiry_str)
            if expiry.normalize() < as_of.normalize():
                continue
            T = max((expiry.normalize() - as_of.normalize()).days / 365.0, 1 / 365.0)

            for strike_str, contracts in strikes.items():
                for c in contracts:
                    strike = float(strike_str)
                    oi     = float(c.get('openInterest', 0) or 0)
                    iv     = float(c.get('volatility', 0) or 0) / 100.0  # Schwab 给的是百分比
                    vol    = float(c.get('totalVolume', 0) or 0)

                    # Schwab 直接给 gamma
                    gamma = float(c.get('gamma', 0) or 0)
                    if gamma <= 0 and iv > 0:
                        gamma = bs_gamma(spot, strike, T, RISK_FREE_RATE, iv)

                    if strike <= 0 or oi <= 0 or gamma <= 0:
                        continue

                    records.append({
                        'as_of':         as_of,
                        'expiry':        expiry,
                        'option_type':   opt_type,
                        'strike':        strike,
                        'gamma':         gamma,
                        'open_interest': oi,
                        'volume':        vol,
                        'iv':            iv,
                        'delta':         float(c.get('delta', 0) or 0),
                    })

    df_out = pd.DataFrame(records)
    print(f'  载入 {len(df_out)} 条期权数据，{df_out["expiry"].nunique() if not df_out.empty else 0} 个到期日')
    return df_out, spot


# ════════════════════════════════════════════════════════════════════
#  GEX 结构计算
# ════════════════════════════════════════════════════════════════════

def compute_gex(df: pd.DataFrame, spot: float) -> dict:
    df = df.copy()
    df['gex']        = df['gamma'] * df['open_interest'] * CONTRACT_SIZE * (spot ** 2) * 0.01
    df['signed_gex'] = np.where(df['option_type'] == 'C', df['gex'], -df['gex'])

    calls = df[df['option_type'] == 'C'].groupby('strike').agg(
        call_gex=('gex', 'sum'), call_oi=('open_interest', 'sum'))
    puts  = df[df['option_type'] == 'P'].groupby('strike').agg(
        put_gex=('gex', 'sum'),  put_oi=('open_interest', 'sum'))

    by_strike              = calls.join(puts, how='outer').fillna(0)
    by_strike['net_gex']   = by_strike['call_gex'] - by_strike['put_gex']
    by_strike['gross_gex'] = by_strike['call_gex'] + by_strike['put_gex']
    by_strike              = by_strike.sort_index()

    call_wall = float(by_strike['call_gex'].idxmax())
    put_wall  = float(by_strike['put_gex'].idxmax())

    cumulative    = by_strike['net_gex'].cumsum()
    balance_score = (2.0 * cumulative - cumulative.iloc[-1]).abs()
    gex_pain      = float(balance_score.idxmin())

    strikes = by_strike.index.to_numpy(dtype=float)
    call_oi = by_strike['call_oi'].to_numpy(dtype=float)
    put_oi  = by_strike['put_oi'].to_numpy(dtype=float)
    payout  = [
        (np.maximum(c - strikes, 0) * call_oi).sum() +
        (np.maximum(strikes - c, 0) * put_oi).sum()
        for c in strikes
    ]
    max_pain = float(strikes[int(np.argmin(payout))])

    signs   = np.sign(by_strike['net_gex'].to_numpy())
    crosses = np.where(signs[:-1] * signs[1:] < 0)[0]
    gamma_flip = None
    if crosses.size > 0:
        nearest    = crosses[np.argmin(np.abs(strikes[crosses] - spot))]
        left, right = strikes[nearest], strikes[nearest + 1]
        lv, rv     = by_strike['net_gex'].iloc[nearest], by_strike['net_gex'].iloc[nearest + 1]
        w          = abs(lv) / (abs(lv) + abs(rv)) if (abs(lv) + abs(rv)) > 0 else 0.5
        gamma_flip = float(left + w * (right - left))

    return {
        'call_wall':  call_wall,
        'put_wall':   put_wall,
        'gex_pain':   gex_pain,
        'max_pain':   max_pain,
        'gamma_flip': gamma_flip,
        'total_gex':  float(by_strike['net_gex'].sum()),
        'by_strike':  by_strike,
    }


# ════════════════════════════════════════════════════════════════════
#  打印报告
# ════════════════════════════════════════════════════════════════════

def print_report(symbol: str, spot: float, levels: dict, position=None):
    tgex   = levels['total_gex']
    gf     = levels['gamma_flip']
    cw, pw = levels['call_wall'], levels['put_wall']
    regime = '做市商多头 Gamma（低波动，倾向压针）' if tgex > 0 else '做市商空头 Gamma（波动放大，趋势延伸）'

    print('\n' + '═' * 54)
    print(f'  {symbol}  GEX 结构报告   {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('═' * 54)
    print(f'  现价        ${spot:.2f}')
    print(f'  Call Wall   ${cw:.1f}    ← 上方最大 Gamma 阻力')
    print(f'  Put Wall    ${pw:.1f}    ← 下方最大 Gamma 支撑')
    print(f'  GEX Pain    ${levels["gex_pain"]:.1f}    ← 做市商净 Gamma 均衡点')
    print(f'  Max Pain    ${levels["max_pain"]:.1f}    ← 期权买方最痛点')
    if gf:
        print(f'  Gamma Flip  ${gf:.1f}    ← 穿越后做市商对冲方向反转')
    print(f'  总净 GEX    ${tgex / 1e9:.3f}B')
    print(f'  市场状态    {regime}')
    print('─' * 54)

    if pw <= spot <= cw:
        print('  📍 现价在墙之间 → 做市商有压针动力，震荡概率高')
    elif spot > cw:
        print('  🔺 突破 Call Wall → 做市商买入对冲，可能加速上涨')
    else:
        print('  🔻 跌破 Put Wall → 做市商卖出对冲，可能加速下跌')

    if position:
        lo, hi, cost = position['lower'], position['upper'], position['cost']
        max_profit   = round((hi - lo - cost) * 100)
        max_loss     = round(cost * 100)

        print('─' * 54)
        print(f'  你的仓位: {symbol} ${lo}/{hi} Call Spread   成本 ${cost:.2f}')

        if spot < lo:
            print(f'  ⚠️  现价 ${spot:.1f}，距下界还差 ${lo - spot:.1f}')
        elif spot > hi:
            print(f'  ✅ 现价 ${spot:.1f} 已超过上界，spread 接近最大价值')
        else:
            pct = (spot - lo) / (hi - lo) * 100
            print(f'  📍 现价在区间内，进度 {pct:.0f}%（${lo} → ${hi}）')

        if lo <= cw <= hi:
            print(f'  ⚠️  Call Wall ${cw:.0f} 在区间内，价格可能在此遇阻')
        elif cw > hi:
            print(f'  ✅ Call Wall ${cw:.0f} 在区间上方，上涨路径未受压制')

        print(f'  最大盈利  ${max_profit}   最大亏损  ${max_loss}')

    print('═' * 54)


# ════════════════════════════════════════════════════════════════════
#  可视化
# ════════════════════════════════════════════════════════════════════

def plot_gex(symbol: str, spot: float, levels: dict, position=None):
    by_strike = levels['by_strike']
    lo_range  = spot * 0.88
    hi_range  = spot * 1.12
    nearby    = by_strike[(by_strike.index >= lo_range) & (by_strike.index <= hi_range)].copy()

    if nearby.empty:
        print('⚠  没有足够的行权价数据用于绘图')
        return

    strikes = nearby.index.to_numpy(dtype=float)
    width   = max(np.diff(strikes).min() * 0.6, 0.3) if len(strikes) > 1 else 1.0

    BG, GREEN, RED, GOLD, WHITE = '#0d1117', '#00d4aa', '#ff4b4b', '#ffd700', '#e6edf3'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), facecolor=BG)
    fig.suptitle(
        f'{symbol}  GEX 结构  |  现价 ${spot:.2f}  |  {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        color=WHITE, fontsize=13, y=0.98,
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.tick_params(colors=WHITE)
        for sp in ax.spines.values():
            sp.set_color('#30363d')

    colors_net = [GREEN if v >= 0 else RED for v in nearby['net_gex']]
    ax1.bar(strikes, nearby['net_gex'] / 1e6, color=colors_net, alpha=0.85, width=width)
    ax1.set_title('净 GEX（绿=做市商多头 / 红=空头）', color=WHITE, fontsize=11, pad=6)
    ax1.set_ylabel('Net GEX ($M)', color=WHITE)
    ax1.axhline(0, color=WHITE, linewidth=0.5, alpha=0.4)

    ax2.bar(strikes,  nearby['call_gex'] / 1e6, color=GREEN, alpha=0.75, width=width, label='Call GEX')
    ax2.bar(strikes, -nearby['put_gex']  / 1e6, color=RED,   alpha=0.75, width=width, label='Put GEX（取反）')
    ax2.set_title('Call GEX vs Put GEX', color=WHITE, fontsize=11, pad=6)
    ax2.set_ylabel('GEX ($M)', color=WHITE)
    ax2.set_xlabel('行权价', color=WHITE)
    ax2.axhline(0, color=WHITE, linewidth=0.5, alpha=0.4)
    ax2.legend(facecolor='#161b22', labelcolor=WHITE, fontsize=9)

    key_levels = [
        (spot,                    WHITE,     '--', f'现价 ${spot:.1f}'),
        (levels['call_wall'],     GREEN,     '-',  f'Call Wall ${levels["call_wall"]:.0f}'),
        (levels['put_wall'],      RED,       '-',  f'Put Wall ${levels["put_wall"]:.0f}'),
        (levels['gex_pain'],      GOLD,      '-',  f'GEX Pain ${levels["gex_pain"]:.0f}'),
        (levels['max_pain'],      '#ff8c00', ':',  f'Max Pain ${levels["max_pain"]:.0f}'),
    ]
    if levels['gamma_flip']:
        key_levels.append(
            (levels['gamma_flip'], '#bf5fff', '--', f'Gamma Flip ${levels["gamma_flip"]:.0f}')
        )

    for price, color, ls, label in key_levels:
        if lo_range <= price <= hi_range:
            ax1.axvline(price, color=color, linestyle=ls, linewidth=1.5, alpha=0.9)
            ax2.axvline(price, color=color, linestyle=ls, linewidth=1.5, alpha=0.9)
            ymin = ax2.get_ylim()[0]
            ax2.text(price + width * 0.3, ymin * 0.85 if ymin < 0 else 0,
                     label, color=color, fontsize=7.5, rotation=90, va='bottom')

    if position:
        lo_p, hi_p = position['lower'], position['upper']
        for ax in (ax1, ax2):
            ax.axvspan(lo_p, hi_p, alpha=0.18, color=GOLD)
        ax1.text((lo_p + hi_p) / 2, ax1.get_ylim()[1] * 0.88,
                 f'你的仓位\n${lo_p}/{hi_p}',
                 color=GOLD, fontsize=8, ha='center', va='top')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = f'{symbol.lower()}_gex.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()
    print(f'\n图表已保存: {out_path}')


# ════════════════════════════════════════════════════════════════════
#  主程序
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else 'NVDA'

    # ── 你的持仓（改这里）──────────────────────────────────────────
    my_position = {
        'lower': 210,
        'upper': 220,
        'cost':  2.68,
    } if symbol == 'NVDA' else None

    print(f'正在拉取 {symbol} 数据...')

    try:
        df, spot = fetch_options(symbol)
    except RuntimeError as e:
        print(f'❌ {e}')
        sys.exit(1)

    if df.empty:
        print('❌ 没有获取到期权数据')
        sys.exit(1)

    levels = compute_gex(df, spot)
    print_report(symbol, spot, levels, my_position)
    print('\n正在生成 GEX 图表...')
    plot_gex(symbol, spot, levels, my_position)
