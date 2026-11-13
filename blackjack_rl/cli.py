"""命令列參數與 RuleConfig / ObsConfig 的轉換。"""


def _decks(s: str):
    """'6' → 6；'4-8' → (4, 8)"""
    if "-" in s:
        lo, hi = s.split("-")
        return int(lo), int(hi)
    return int(s)


def _pair(s: str):
    lo, hi = s.split("-")
    return float(lo), float(hi)


def add_rule_args(p):
    g = p.add_argument_group("遊戲規則")
    g.add_argument("--decks", type=_decks, default=6, help="副數，例如 6 或 4-8（每次洗牌隨機）")
    g.add_argument("--penetration", type=_pair, default=(0.65, 0.80), help="cut card 位置比例，例如 0.65-0.80")
    g.add_argument("--burn", type=int, default=1, help="洗牌後燒牌張數")
    g.add_argument("--enhc", action="store_true", help="歐式規則（無暗牌、無 peek）")
    g.add_argument("--h17", action="store_true", help="莊家軟 17 要補牌")
    g.add_argument("--bj-payout", type=float, default=1.5, help="Blackjack 賠率，3:2=1.5，6:5=1.2")
    g.add_argument("--double-rule", default="any", choices=["any", "9-11", "10-11"])
    g.add_argument("--no-das", action="store_true", help="分牌後不可 double")
    g.add_argument("--max-hands", type=int, default=4)
    g.add_argument("--rsa", action="store_true", help="允許再分 A")
    g.add_argument("--hsa", action="store_true", help="分 A 後可繼續抽牌")
    g.add_argument("--no-surrender", action="store_true")
    g.add_argument("--no-betting", action="store_true", help="關閉下注決策（固定 1 倍注）")
    g.add_argument("--bet-sizes", default="1,2,4,8", help="可選下注倍數，逗號分隔")
    g.add_argument("--shoe-info", default="seen_counts", choices=["none", "seen_counts", "hilo"],
                   help="state 可包含的牌堆資訊（一律不含副數與剩餘張數）")
    g.add_argument("--total-encoding", default="onehot", choices=["onehot", "scalar"],
                   help="手牌點數編碼")


def rules_from_args(a):
    from .config import RuleConfig
    return RuleConfig(
        n_decks=a.decks, penetration=a.penetration, burn_cards=a.burn,
        dealer_peek=not a.enhc, dealer_hits_soft17=a.h17, blackjack_payout=a.bj_payout,
        double_rule=a.double_rule, double_after_split=not a.no_das, max_hands=a.max_hands,
        resplit_aces=a.rsa, hit_split_aces=a.hsa, late_surrender=not a.no_surrender,
        enable_betting=not a.no_betting,
        bet_sizes=tuple(float(x) for x in a.bet_sizes.split(",")),
    )


def obs_from_args(a):
    from .config import ObsConfig
    return ObsConfig(shoe_info=a.shoe_info, total_encoding=a.total_encoding)
