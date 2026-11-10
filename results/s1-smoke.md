# S1　轉接與冒煙測試

日期：2026-11-10。Modal L40S，一次執行，容器 264 秒，估 $0.14（預算 $2）。

## 門檻（跑之前寫好，見交接文件 S1）

| 門檻 | 結果 | 判定 |
|---|---|---|
| option mass 中位數 ≥ 0.9，且 option mass < 0.5 的判斷比例 ≤ 2% | 中位數 **1.000**，第 1 百分位 0.9994，最小值 0.989；低於 0.5 的 **0 筆**（共 3,289 筆判斷） | 過（第一次就過，沒有修模板） |
| 推估每 100 題的成本 ≤ $1.2 | **$0.37**（不含冷啟動）／**$0.47**（含一次冷啟動） | 過 |
| 10 題全部跑完，沒有 context 超長的錯誤 | 10 / 10 完成；最長的 prompt 3,996 token（context 16,384） | 過 |

## 做了什麼

- `adapters/jevlike_backend.py`：照上游本地 reranker 的介面寫的 `jevlike` 後端，透過 HTTP 呼叫同一個容器裡的 SGLang `/generate`。送 `input_ids`、自己帶 `<bos>`、`max_new_tokens=1`，讀 A、B 兩個字母 token 的 logprob。分數是 P(A) / (P(A) + P(B))，同時記下 option mass。逐段判斷，不打包。
- `patches/0001-jevlike-backend.patch`：對上游的改動只有兩處，`--reranker` 多一個 `jevlike` 選項，以及一個選後端的工廠函式。submodule 本身沒動，patch 在 Modal image 建置時套到副本上。
- `modal_baseline.py`：SGLang（Gemma 4 26B-A4B FP8，context 16,384，radix cache 預設開啟，不開 MTP）和上游的 `run_sag2_benchmark.py --mode local --reranker jevlike` 跑在同一個 L40S 容器。App 名稱 `sag2poc-a-baseline`，結果寫到 Volume `sag2poc-a-results`。權重沿用 jevlike 既有的 Volume `gb10-decide-hf`（只讀）。
- 模板是 jevlike 的 `GEMMA4_TEMPLATE_NOTHINK_BOS`，一字未改。新增的題型沿用 jevlike v8 英文題的寫法（`[State]`、`[Question]`、英文 system 句）。
- prompt 順序：模板開頭 → 原問題 → 已有證據 → 候選段落 → 問句與選項（A. Yes／B. No）。問句直接用上游本地模式的證據評分提示詞，它本來就是肯定問句。
- 每一輪先送一段把共用前綴暖起來，其餘的再以每批 64 段送出。

送上 GPU 之前，先在本機用假的 SGLang 伺服器把整條上游迴圈乾跑過一次，確認 patch 和轉接層接得上。

## 量到的數字（MuSiQue dev 前 10 題，兩輪擴展，題目並行 16）

| 項目 | 數字 |
|---|---:|
| 每題判斷次數 | 中位數 **334**，第 90 百分位 337 |
| 每題輸入 token | 平均 571,290（中位數 534,554） |
| 其中快取命中 | 82.5% |
| 每題實際要算的 token（沒命中快取） | 約 100,073 |
| 每秒 prefill token（沒命中快取的部分） | 約 14,500 |
| 每秒 prefill token（含快取命中） | 約 83,000 |
| 每秒判斷次數 | 47.8 |
| 每題 GPU 秒數（整段牆鐘 ÷ 題數） | 6.9 秒 |
| 冷啟動（啟動 SGLang 到 health 通過） | 175 秒 |
| GPU 記憶體（載入後） | 37,969 MiB／46,068 MiB |
| 10 題的牆鐘時間 | 68.8 秒 |

- **每題判斷次數的中位數是 334，超過 150 的標示線。** 這是 SAG2 迴圈本身的形狀（三輪、每輪約 100 到 125 段），會直接影響 GB10 上的延遲。
- 容器裡第一輪 BM25 的逐題分數，和 S0 在本機用上游程式跑的完全相同（10 題零差異），詞彙層沒有因為換環境而改變。
- 這 10 題的分數只當作「接得上」的證據，樣本太小不能下結論：B0 R@5 83.3、AE@10 70.0；同 10 題 BM25 R@5 35.8，SAG2＋Jev 85.0。

## 成本推估（後面各段用這個數字估）

- **每 100 題約 $0.37**（兩輪擴展，L40S 以 $1.95／小時計），每次執行另加一次冷啟動約 $0.10（175 秒）加容器收尾。
- 只判一輪（S2）每題約 100 次判斷、沒有長前綴，預估每 100 題不到 $0.15。
- S3（dev 400 題，兩輪擴展）預估約 $1.6，在 $4 預算內。

## 給 Session B 的實測備忘（S5 寫交接時用）

- SGLang 版本 `0.5.12.post2.dev932`，image `lmsysorg/sglang:gemma4-mtp`，L40S 要帶 jevlike 的 fused-MoE Triton 設定檔。
- Modal 的 Python client 在這個環境要裝 `modal[api-proxy-support]` 才連得上。
- 用 HF `tokenizers` 直接讀模型目錄的 `tokenizer.json`，文字裡的 `<bos>` 會被對成特殊 token（id 2）；A、B 各是單一 token。
- 長前綴（約 3,000 token 的已有證據）之下 option mass 仍然接近 1，讀法在長段落上是穩的。

## 結論

**繼續**做 S2。三項門檻都通過。
