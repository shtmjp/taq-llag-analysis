# `pair_summary.csv` 説明文書

## 1. この表が表しているもの

`pair_summary.csv` は、日付、銘柄、trade 側の取引所コード、quote 側の取引所コードの組ごとに、
trade event process と quote event process の時間差構造を要約した表である。

1 行は 1 つの組

- `run_id`
- `date_yyyymmdd`
- `symbol`
- `trade_exchange`
- `quote_exchange`

に対応する。

ここで lag は

`d = t_quote - t_trade`

で定義される。したがって

- `d < 0` は quote の方が trade より先
- `d > 0` は quote の方が trade より後

を意味する。時間の単位はすべて秒である。

## 2. 行が作られる条件

各日付・各銘柄について、trade 側と quote 側の event time 列から、取引所別の event time 列を作る。

- trade 側は 1 つの取引所コードを固定し、同じ timestamp をまとめた後、event time に変換する。
- quote 側も 1 つの取引所コードを固定し、同じ timestamp をまとめた後、event time に変換する。
- 観測窓は `09:45:00` から `15:45:00` までである。
- 候補ペアは `trade_exchange != quote_exchange` を満たす組だけである。

したがって、この表は「すべての銘柄について固定の 1 ペアだけを持つ表」ではない。
同じ銘柄・同じ日付でも、取引所コードの組が違えば別の行になる。

## 3. `status` の意味

`status` は、その行で解析が最後まで完了したかを示す。

- `ok`
  - 解析が正常終了した。
  - `bandwidth`、`mode_count`、`closest_mode_to_zero_sec`、`cross_k_*` が埋まる。
- `skipped_min_events`
  - trade 側または quote 側のどちらかで、観測窓内の event 数が最小必要数に達しなかった。
  - この run では最小必要数は 100 である。
  - 解析指標列は空欄になる。
- `skipped_not_simple`
  - trade 側と quote 側が完全に同じ event time を 1 つ以上共有していたため、解析を行わなかった。
  - この場合 `n_shared_event_times`、`error_type`、`error_message` は埋まるが、解析指標列は空欄になる。
- `error`
  - 解析中に例外が発生したことを表す予約値である。
  - この run では `error` 行は含まれていないが、スキーマ上は許される。

## 4. 列ごとの説明

| 列名 | 意味 | 単位・型の目安 | 空欄になる条件 |
|---|---|---|---|
| `run_id` | その行が属する解析 run の識別子。 | 文字列 | なし |
| `date_yyyymmdd` | 対象営業日。`YYYYMMDD` 形式。 | 文字列 | なし |
| `symbol` | 銘柄コード。 | 文字列 | なし |
| `trade_exchange` | trade 側 process の取引所コード。 | 1 文字の文字列 | なし |
| `quote_exchange` | quote 側 process の取引所コード。 | 1 文字の文字列 | なし |
| `status` | 解析結果の状態。 | 文字列 | なし |
| `n_trade_events` | 観測窓内で使われた trade event 数。取引所コードを固定し、同一 timestamp をまとめた後の個数。 | 整数 | なし |
| `n_quote_events` | 観測窓内で使われた quote event 数。取引所コードを固定し、同一 timestamp をまとめた後の個数。 | 整数 | なし |
| `n_shared_event_times` | trade 側と quote 側が完全に同じ event time を共有した個数。 | 整数 | `skipped_min_events` のとき空欄 |
| `bandwidth` | CPCF の mode 探索に使われた bandwidth。候補集合から選ばれる。 | 秒、実数 | `status != ok` のとき空欄 |
| `mode_count` | 探索区間内で CPCF 推定値が最大になる lag の個数。 | 整数 | `status != ok` のとき空欄 |
| `closest_mode_to_zero_sec` | 最大点が複数あるとき、その中で 0 秒に最も近い lag。 | 秒、実数 | `status != ok` のとき空欄 |
| `elapsed_sec` | その行の解析処理に要した時間。 | 秒、実数 | `skipped_min_events` のとき空欄 |
| `cross_k_neg_1e3_1e4` | lag 区間 `[-1e-3, -1e-4]` 秒で計算した cross-K 統計量。 | 実数 | `status != ok` のとき空欄 |
| `cross_k_neg_1e4_1e5` | lag 区間 `[-1e-4, -1e-5]` 秒で計算した cross-K 統計量。 | 実数 | `status != ok` のとき空欄 |
| `cross_k_neg_1e5_0` | lag 区間 `[-1e-5, 0]` 秒で計算した cross-K 統計量。 | 実数 | `status != ok` のとき空欄 |
| `cross_k_pos_0_1e5` | lag 区間 `[0, 1e-5]` 秒で計算した cross-K 統計量。 | 実数 | `status != ok` のとき空欄 |
| `cross_k_pos_1e5_1e4` | lag 区間 `[1e-5, 1e-4]` 秒で計算した cross-K 統計量。 | 実数 | `status != ok` のとき空欄 |
| `cross_k_pos_1e4_1e3` | lag 区間 `[1e-4, 1e-3]` 秒で計算した cross-K 統計量。 | 実数 | `status != ok` のとき空欄 |
| `error_type` | `skipped_not_simple` または `error` の原因分類。 | 文字列 | `ok` と `skipped_min_events` では空欄 |
| `error_message` | `skipped_not_simple` または `error` の補足説明。 | 文字列 | `ok` と `skipped_min_events` では空欄 |

## 5. `mode_count` と `closest_mode_to_zero_sec` の解釈

この表の mode は「もっとも頻度が高い timestamp」ではない。
trade 側 process と quote 側 process の lag 構造に対して作られた CPCF
（cross pair correlation function）推定値の最大点である。

探索区間を `U = [-1, 1]` 秒とすると、

- `mode_count` は `U` の中で最大点になった lag の個数
- `closest_mode_to_zero_sec` はその最大点集合のうち `|d|` が最小の lag

を表す。

したがって `closest_mode_to_zero_sec` は

- 平均 lag ではない
- 中央値 lag ではない
- 個々の event 差分の最近傍値でもない

という点に注意が必要である。

## 6. `cross_k_*` 列の読み方

`cross_k_*` は、指定した lag 区間ごとに計算した cross-K 統計量である。
これは raw count ではなく、lag 区間内の cross-event 集中度を要約する統計量として読むべき列である。

列名の `1e3`, `1e4`, `1e5` は、それぞれ

- `1e-3` 秒 = 1 ms
- `1e-4` 秒 = 0.1 ms
- `1e-5` 秒 = 0.01 ms = 10 us

を表す略記である。

各列が表す lag 区間は次のとおりである。

| 列名 | lag 区間（秒） | lag 区間（時間換算） | 向き |
|---|---|---|---|
| `cross_k_neg_1e3_1e4` | `[-1e-3, -1e-4]` | `[-1 ms, -0.1 ms]` | quote が 0.1 ms から 1 ms 先 |
| `cross_k_neg_1e4_1e5` | `[-1e-4, -1e-5]` | `[-0.1 ms, -0.01 ms]` | quote が 10 us から 100 us 先 |
| `cross_k_neg_1e5_0` | `[-1e-5, 0]` | `[-0.01 ms, 0]` | quote が最大 10 us 先 |
| `cross_k_pos_0_1e5` | `[0, 1e-5]` | `[0, 0.01 ms]` | quote が最大 10 us 後 |
| `cross_k_pos_1e5_1e4` | `[1e-5, 1e-4]` | `[0.01 ms, 0.1 ms]` | quote が 10 us から 100 us 後 |
| `cross_k_pos_1e4_1e3` | `[1e-4, 1e-3]` | `[0.1 ms, 1 ms]` | quote が 0.1 ms から 1 ms 後 |

## 7. 欠損値の読み方

空欄は「0」ではなく、「その列がその status では定義されていない」ことを意味する。

典型的には次のように読む。

- `status = skipped_min_events`
  - event 数は分かるが、解析指標は未計算である。
- `status = skipped_not_simple`
  - event 数と共有 timestamp 数は分かるが、解析指標は未計算である。
- `status = ok`
  - 主要な解析指標が埋まる。

## 8. この表だけで分かることと、分からないこと

この表だけで分かること:

- どの日付・銘柄・取引所ペアが解析対象になったか
- 解析が成功したか、どの理由でスキップされたか
- 観測窓内の event 数
- 代表的な mode と lag-window ごとの cross-K 要約値

この表だけでは分からないこと:

- 最大点の全リスト
- 個々の trade event と quote event の生時刻
- exchange code の正式名称
- 統計量の値を生んだ元の event 列全体

## 9. 要約

`pair_summary.csv` は、日付・銘柄・trade 取引所・quote 取引所の各組について、

- event 数
- 解析の成否
- 共有 timestamp の有無
- CPCF に基づく代表的 lag
- 複数の微小 lag 区間における cross-K 統計量

を 1 行にまとめた要約表である。
