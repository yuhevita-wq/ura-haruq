# 裏ハルキュー（URA-HARUQ）

## 【正式採用】裏ハルキュー6点方式の参加選別 V2（2026-10-26）

2026年10月26日、ユーザーの明示指示により、**固定の3連単6点ルール（V1）に実力差フィルタを組み込んだV2**を正式採用。

- コード：`ura/frozen_v1.py`（**旧V1と同一コード**）、`ura/participant_v2.py`（その外側の正式参加判定）
- 予測順位は、実証に使った **独立ウォークフォワード順位モデル**を使うこと（[実装側](https://github.com/yuhevita-wq/keirin-ai-predictor/tree/ura-haruq/independent-train-20261024/ura_haruq_independent)）。本家の古いスコアに置き換えない
- V1の資格：7車・2-2-2-1・独立スコア差[0,1)・上位3位の同ライン2車を両順・別線の予測5〜7位を3着＝**3連単6点**
- **新しい参加条件：7車の競走得点（出走表）の最大値－最小値が7.1以下**。欠損したら見送り。
- 1点100円、該当時は600円。条件未達は0円。
- 旧追加条件「3着候補の平均競走得点95.8767超」は廃止。級班によるA級排除を回避する
- `python -m ura.participant_v2 --input racecards_and_independent_ranker.json --out decisions.json` で、保存済みの独立AI順位・出走表ライン・競走得点から本番参加判定を実行する
- 人が読める出力に `base_tickets`（旧V1候補）と `tickets`（V2の参加対象買い目）を分離。発走前に確認し、履歴は別期間で追跡

**既存の確率校正AI `ura/core.py`・`ura/train.py`・`ura/predict.py` の処理は書き換えない。** あちらは候補確率を列挙する別の研究系統。両者の学習済みモデルや購入条件を黙って混用しない。

既存3,501レースから適用した2024/25/26(1〜9月)の**採用V2のバックテスト**は313/291/209レース、回収率223.16%/212.53%/223.45%。ユーザーが既知の結果で選別を改良したため新たなブラインド検証ではない。発走前44特徴量の監査・取引時点の3連単オッズ照合・未来期間の確認は今後の研究課題。

[正式仕様](https://github.com/yuhevita-wq/keirin-ai-predictor/blob/ura-haruq/independent-train-20261024/ura_haruq_independent/ADOPTED_V2_POLICY_20261026.md) ／ [3,501レースとの再照合・CI成功記録](https://github.com/yuhevita-wq/keirin-ai-predictor/blob/ura-haruq/independent-train-20261024/ura_haruq_independent/ADOPTED_V2_VERIFICATION.json)

---


本家ハルキューとは**完全に別のGitHubリポジトリ**です。本家の固定AI、44特徴量、判定条件、買い目、購入金額、Webサイトを変更しません。

## 研究対象：③ LOWレース × 低評価選手

- **LOWレース**：本家の固定AIスコア差（評価1位−2位）が1.00未満
- **低評価選手**：7車ならAI評価5〜7位。9車なら6〜9位
- **3連単候補**：1〜3着のどこかに下位評価選手を含む全順列
- 7車のLOWレースでは186候補。これは購入点数ではなく研究対象数です。
- 高配当・期待値を示すには発走前オッズと実際の着順を突き合わせた別の未学習期間検証が必要です。

## 独立AIの実装

- `ura/core.py`：3連単の全順列確率、低信頼×低順位抽出、順位学習の目的関数
- `ura/train.py`：本家モデルの保存済みバイナリと大量の実結果を利用する、裏専用の時系列学習
- `ura/predict.py`：裏学習済みモデル＋本家のスコアJSONを入力して裏3連単全候補を提示。オッズCSVがあれば参考期待値を計算するが、自動購入はしない
- `features/`、`models/`、`db/`、`config.py`：本家2026年3月21日の固定ソースから**同一内容で複製**
- `.github/workflows/ura-verify.yml`：テストと固定特徴量の一致検査

本家の固定モデル SHA256：`e70690f4538b77e927a042e7ae2db312f8a0a4f16bf72def37e7ccad3568c193`。

## 時系列分割

| 期間 | 役割 |
|---|---|
| 2025年7月〜2026年3月 | 裏AIの学習 |
| 2026年4月〜5月 | 検証 |
| 2026年6月〜9月 | 完全な未学習検証 |

学習対象はLOWレースのみ。元の44特徴量LightGBMは**再学習しません**。順序確率の係数だけを独立学習します。

## 既存データの利用

既に本家 `yuhevita-wq/haruq` のGitHub Releasesに保存済み。

- `haruqube-20260321-flagship-v1` の固定モデル
- `haruqube-20260321-history-cache-live` の暗号化SQLite履歴

これらは取得・読込のみ。新しいデータ収集は不要です。学習には暗号化データの復号が必要なため、裏専用リポジトリのGitHub Actionsにシークレット `COLLECTOR_PASSPHRASE` を設定する必要があります。**チャットに復号鍵を送らないでください。**

**2026-10-24 実データ学習成功。** GitHub Actions [実行 #40047635269](https://github.com/yuhevita-wq/ura-haruq/actions/runs/40047635269) で暗号化履歴の復号、元モデルの照合、裏専用の時系列学習とArtifact保存が成功しました。学習12,163件、検証2,537件、未学習テスト5,505件。テスト対数損失は基準4.6941274から裏学習後4.6497945へ改善。モデルは [artifacts/ura_ai_v1.json](artifacts/ura_ai_v1.json) に永続保存しました。利益・期待値は未検証です。

## ローカル動作確認

1. 依存パッケージを `pip install -r requirements.txt` で導入
2. `python -m unittest discover -s tests -v` で数理テスト
3. 適切に復号済みの `db/keirin.db` とハッシュ確認済み固定モデルがある隔離環境で `python -m ura.train --model モデルへのパス --out ura_outputs`
4. 学習済み成果物が実際に生成された後、`python -m ura.predict --prediction 元スコアJSON --trained ura_outputs/ura_ai_v1.json --out ura_prediction.json`

**現状の限界**：元の過去学習用特徴量には結果データ由来のオッズ・人気が含まれています。発走前取得時刻との一致が監査されるまでは、学習済みであっても実運用の回収率・期待値を保証できません。


---

## 【新規組み込み】独立フォーメーション理論2型（2026-10-26）

**追加コード**：ura/market_shapes_v1.py / ura/integrated_signals.py。
今回限りの明示許可により、**既存の固定V1・採用V2の予測、6点、参加判定、購入金額は変更せず**、新しい2型の研究候補出力を追加。独立AIの順位入力は不要です。

全7車・全210点の3連単オッズ、レース前の予想並び（例：1-7/5-4/6-2/3）、級班を使用。全オッズの逆数を正規化し、各車の1着市場支持をラインごとに合算。上位3ラインをL1・L2・L3とし、L1先頭=A、番手=a、L3先頭=B、番手=bとします。L2は飛び越します。

| 新理論 | 3連単フォーメーション | 合格条件 | 参考点数 |
| --- | --- | --- | --- |
| 二線飛び越しクロス型 | A-B-a / A-B-b / B-A-a / B-A-b | 4点の幾何平均オッズが200倍以上400倍未満 | 4点（100円ずつで400円） |
| 敵線残り二人型 | A-B-b / B-A-a | A級1・2班かつ正規化した2点の市場支持合計が0.010以上0.025未満 | 2点（100円ずつで200円） |

市場順位は人数順でも競走得点順でもなく、全3連単から求めた1着支持のライン合計順。同点では先頭の市場支持、車番順。予想並びの先頭・番手を使用し、結果や従来V1/V2のスコアで並べ替えません。

### 統合出力の分離

- independent_new_shapes.candidates：2型の成立・不成立、条件値、買い目、オッズ、参考投資金額。
- independent_new_shapes.overlap_tickets：2型に同時該当した重複車券。unique_research_tickets：重複除去後の一覧。重複購入は自動実行しません。
- adopted_v2：旧ura/participant_v2.pyの結果をそのまま掲載。official_ticketsとofficial_decisionはV2のみ。
- 新型はresearch_only=true・purchase_authorized=false。正式V2購入判定とは区別します。

実行：

    python -m ura.integrated_signals --input races.json --out ura_integrated_signals.json

入力例（全210オッズのうち省略した説明用の例で、そのままでは実行不可）：

    {
      "races": [{
        "race_id": "sample",
        "formation": "1-7/5-4/6-2/3",
        "race_class": "A1/A2",
        "trifecta_odds": {"1-2-3": 100.0}
      }]
    }

実際には全210の異なる3連単組番を省略せず指定します。欠損・不正オッズを許容しません。旧V2の評価も必要な場合、predicted_bikes_by_rank、line_groups_by_bike、score_gap、competition_scores_by_bikeを各レースに追加してください。V2入力がなければ旧V2はNOT_EVALUATED、新型入力がなければ新型はNOT_EVALUATEDです。

### 実証上の留保

2025年下期〜2026年上期の探索は**確定オッズ**で行ったもので、選択バイアスがあります。未見の将来期間・締切前オッズで利益は未実証。新型の自動購入判定は有効化していません。前向き検証では購入判断時点のオッズ・予想並びの保存と結果との分離が必要です。既存Webおよび本家ハルキューは変更していません。
