# B0 歸因補測 — 交接文件 v0.1

日期：2026-11-11。接續 `SAG2POC-handoff.md` v0.5 與 Taguri（`clarencechien/taguri`）的全案結論。

> **給誰做**：只能由 **Session A 這一側**的 session 做（掛 `sag2poc-baseline`，可以讀上游 SAG2 的程式）。
> **不能派給 Session B**：兩項都要執行、其中一項還要改接上游的程式。Session B 一碰就破壞 clean-room，Taguri 之後如果繼續開發，獨立實作的紀錄就不成立了。
>
> 每一項都**先寫好判定規則再跑**（規則已寫在下面，不要改），跑完填結論，commit + push。

## 0. 要回答的問題

Taguri 有兩個地方落後，目前都分不出原因：

| # | 現象 | 分不出什麼 |
|---|---|---|
| C1 | 中文比英文低約 13 分（dev-zh 100 題，R@5：英 80.25、繁 66.33、簡 67.58） | 是 Taguri 的流程在中文上特別弱，還是任何迴圈配 jevlike 在這份資料上都會掉 |
| C2 | EnterpriseRAG 的「專案相關」「完整性」幾乎沒進步（文件 R@10：67.92、49.84；上游紀錄的 SAG2＋Jev 是 83.58、76.24） | 落差來自迴圈的設計，還是來自判官（jevlike 對 Jev） |

兩項都用 **B0**（上游 SAG2 迴圈＋jevlike，`adapters/jevlike_backend.py`）當對照。B0 和 Taguri 用同一顆判官，和 SAG2＋Jev 用同一個迴圈，所以它可以把「迴圈」和「判官」分開。

這兩項回答的是「怪誰」，**不是「能不能用」**。不論結果如何，下一步都還是 GB10 的延遲實測和真實繁中資料。

## 1. 共同規則

- `SAG2POC-handoff.md` §3 的規則全部照舊：不呼叫 Jev；只用公開題組；金鑰只走環境變數；每次判斷記原始 log P(A)、log P(B)；每題判斷次數、token、GPU 秒都要記。
- B0 保持上游原樣：參數用上游預設值，提示詞沿用上游本地模式的英文提示詞，**不校準、不為中文改任何東西**。
- Modal：app 與結果 Volume 沿用 `sag2poc-a-baseline`、`sag2poc-a-results`。
- **Session B 的 Volume `taguri-b-results` 只讀不寫。** 用 `modal volume get` 取資料，或掛上去讀；不要 commit。
- **往 Taguri 那邊回寫的東西，限制和 §2.5 相同**：只能是逐題分數（題號＋指標）和只有數字與結論的短說明。不寫上游的提示詞、參數值、程式，也不寫上游在某一步怎麼做。回寫要 Clarence 點頭後才做，這份文件的預設是**結果只留在 `sag2poc-baseline`**。
- 預算上限合計 **$8**（C1 $2、C2 $6）。全案目前估約 $29.5（上限 $60，未核對帳單）。預估會超過該項上限就先停下來回報。

## 2. C1　B0 在 dev-zh（上限 $2）

### 資料（都已存在）

| 東西 | 位置 |
|---|---|
| 繁中語料與問題（重翻後的版本） | Volume `taguri-b-results`：`data/musique-zhtw2/`（`corpus.jsonl`、`questions.jsonl`、`splits.json`，切分名 `dev_zh`，100 題） |
| 簡中版 | 同上，`data/musique-zhcn2/` |
| 英文同一批 100 題的 B0 | 已經有：`reference/musique/b0.jsonl`（dev-zh 就是 MuSiQue dev 的前 100 題），各階段在 `b0_h0.jsonl`、`b0_h1.jsonl`；原始紀錄在 `sag2poc-a-results` 的 `s3-b0/` |
| Taguri 在這 100 題的三語結果 | Volume `taguri-b-results`：`s7d/` 底下（先 `modal volume ls` 找到英、繁、簡三組 N1 與 N0 的 `run.jsonl`） |

金標段落的 `pid` 在三種語言之間不變。

### 做什麼

1. 把繁中、簡中兩份資料各轉成上游讀的格式（做法同 `tools/convert_multihop_rag.py` 的後半段：檔名用上游接受的題組名，放在各自的資料夾、各自的索引快取）。只做格式轉換。
2. 用 `tools/run_upstream_bm25.py` 跑上游的純 BM25（CPU），繁中、簡中各一次。
3. 用 `modal_baseline.py` 跑 B0 兩輪擴展，繁中、簡中各 100 題，放在同一個容器裡依序跑（`--dataset-dir` 指到轉好的資料）。
4. 用 `tools/analyze_loop.py` 算最終與各階段分數、校準數字。
5. 用 `evalkit` 對 Taguri 的三語 `run.jsonl` 重新計分，取得逐題分數。

預估：200 題 × 每 100 題 $0.37，加一次冷啟動，約 $1。

### 要報告的數字（同一批 100 題，逐題配對）

| 數字 | 用途 |
|---|---|
| B0 繁中、簡中的 R@5／R@10／AE@10，以及 B0 繁中 − B0 英文、簡中 − 英文 | B0 自己掉多少 |
| 上游 BM25 繁中 − 英文 | 對照 Taguri 詞彙層的 −7.50 |
| **D ＝（Taguri 繁中 − Taguri 英文）−（B0 繁中 − B0 英文）**，配對 bootstrap 的 95% 信賴區間；簡中同樣算一次 | 主要判定用 |
| B0 繁中 − B0 簡中 | 上游沒有繁簡正規化，看它值多少（Taguri 是 −1.25） |
| 各階段金標判「是」的比例、非證據判「是」的比例，中文對英文 | 對照 Taguri 的 52% 對 65% |
| 每題判斷次數、輸入 token、最長 prompt | |

### 判定規則（先寫好，用繁中的 D，R@5）

| 結果 | 判定 |
|---|---|
| D 的信賴區間上界 < 0 | **Taguri 在中文上多掉了一塊**，大小約 |D|。要回頭看 Taguri 的中文候選詞與詞彙層 |
| D 的信賴區間包含 0，而且 B0 繁中 − 英文 ≤ −8 | **兩套迴圈都掉**。原因在判官或這份翻譯資料，不在 Taguri 的設計。不必為這 13 分回頭改 Taguri |
| D 的信賴區間包含 0，而且 B0 繁中 − 英文 > −8 | **分不出來**（100 題的標準誤約 ±4 分），照實寫 |
| D 的信賴區間下界 > 0 | Taguri 在中文上掉得比 B0 少。註明 Taguri 的繁簡正規化與中文設定是有用的 |

再用校準數字把「兩套都掉」拆成兩種情況，只描述、不設門檻：

- B0 的金標判「是」比例在中文也明顯低於英文（英文 H1、H2 約 60%）→ 傾向判官讀中文（或讀這份譯文）比較弱。
- B0 的金標判「是」比例和英文相近，但進到候選的金標變少 → 傾向檢索這一段（斷詞、譯名不一致）。

### 解讀時要注意（寫進結論）

- **提示詞語言不同**：B0 用英文提示詞讀中文段落，Taguri 用中文提示詞。這個差別混在 D 裡面，分不開。
- dev-zh 是機器翻譯的，段落譯文有已知缺陷（`results/s7b-dev-zh.md`，在 taguri repo）。兩套迴圈吃的是同一份資料，所以 D 不受它影響；但「B0 掉多少」這個絕對值會受影響。
- 只有 100 題。

**結論**：＿＿

## 3. C2　EnterpriseRAG 的「專案相關」「完整性」（上限 $6）

分兩步。**第 0 步不用 GPU，先做；它的結果可能讓第 1 步變得沒有必要。**

### 第 0 步：把既有的紀錄配對起來（CPU，$0）

Session B 寫報告時沒有上游的逐題紀錄，所以把 SAG2＋Jev 的數字當成「只能參照、不能配對」。Session A 這一側有：

| 東西 | 位置 |
|---|---|
| 上游 SAG2＋Jev 與上游 BM25 的逐題紀錄（470 題，含題型、金標文件、兩種方法各自撈到的文件編號） | `vendor/SAG-Benchmark/sag2/results/enterprise/questions.jsonl` |
| Taguri 的逐題結果（N0、N1、N1＋EG2） | Volume `taguri-b-results`：`s10/enterprise-N0/run.jsonl`、`s10/enterprise-N1/run.jsonl`、`s10/enterprise-N1+EG2/run.jsonl` |
| Taguri 用的題目與文件編號 | Volume `taguri-b-results`：`data/enterprise/questions.jsonl`、`MANIFEST.json` |

做什麼：

1. 確認兩邊的題號和**文件編號是同一套**（都應該是題組官方的編號）。抽 20 題比對金標文件。對不上就先查清楚怎麼對應，對應不了就在結論寫明，第 0 步到此為止。
2. 用 `evalkit` 以文件為單位，對兩邊的結果重新計分（上游那份先確認重算的數字和它自己記的一致）。
3. 逐題配對：Taguri − SAG2＋Jev、Taguri − 上游 BM25、Taguri 的 N0 − 上游 BM25。報整體 470 題和八個題型，R@10，附 95% 信賴區間。

指標以文件為單位，兩邊分段方式不同不影響配對。

**第 0 步的判定（先寫好）**

| 「專案相關」＋「完整性」60 題上，Taguri − SAG2＋Jev 的 ΔR@10 | 接下來 |
|---|---|
| 信賴區間上界 < 0 | 落差是真的。做第 1 步 |
| 信賴區間包含 0 | 落差在配對之後不成立（原本只是分段方式或抽樣的差別）。**不做第 1 步**，結論寫明 |

### 第 1 步：B0 跑這 60 題

這一步的工作量比 C1 大，動手前先確認四件事，任何一件做不到就停下來回報：

1. **上游的企業題組入口只接 Jev，沒有本地模式。** 要另外寫一個 patch（`patches/0002-enterprise-local-scorer.patch`），把判官換成既有的 jevlike 後端，其餘原封不動。不修改 submodule 本身。先用本機的假 SGLang 伺服器乾跑兩題。
2. **上游的索引要自己建。** 題組官方的資料夾（`documents/`、`questions.jsonl`、`download_manifest.json`）要從 Hugging Face `onyx-dot-app/EnterpriseRAG-Bench` 下載（記下 revision；Session B 用的 revision 在它 Volume 的 `data/enterprise/MANIFEST.json`，用同一個）。51 萬份文件，用上游的建索引腳本在 Modal 的 CPU 函式裡建，存到 `sag2poc-a-results`。建置時間和檔案大小要記。
3. **索引要和上游的對得上。** 建好後不呼叫判官，先比對：我們的索引跑出來的 BM25 前十名，和上游紀錄裡的 BM25 前十名，60 題逐題比。大致一致（R@10 相差 1 分以內）才往下跑；差很多代表題組版本或索引版本不同，先查。
4. **context 長度。** 企業文件的段落比 MuSiQue 長很多，帶已有證據的 prompt 可能超過 16,384 token。轉接層遇到超長會直接報錯、不截斷，這是對的（B0 不截斷）。先跑 5 題看最長的 prompt；超過就把伺服器的 context 加大再跑，並記下 GPU 記憶體夠不夠。加大後仍然放不下，就停下來回報。

跑之前用那 5 題估成本。預估（很粗）：每題判斷次數和多跳題組相同，但每次判斷的 token 是 MuSiQue 的好幾倍，60 題約 $1 到 $3，加上建索引的 CPU 時間與冷啟動。

### 要報告的數字（60 題；「專案相關」40 題和「完整性」20 題分開報，也報合計）

文件層級的 R@10，逐題配對，95% 信賴區間：

| 比較 | 意思 |
|---|---|
| B0 − SAG2＋Jev | 同一個迴圈，只差判官 |
| Taguri − B0 | 同一顆判官，只差迴圈 |
| B0 − 上游 BM25、Taguri − Taguri 的 N0 | 各自的判官迴圈比自己的詞彙層多了多少 |

另外記：每題判斷次數、輸入 token、最長 prompt、每題時間裡等判官和做檢索各占多少、花費。

### 判定規則（先寫好，用 60 題合計的 ΔR@10）

| B0 − SAG2＋Jev | Taguri − B0 | 判定 |
|---|---|---|
| 信賴區間包含 0，或下界 > −5 | 信賴區間上界 < 0 | **落差在迴圈的設計。** Taguri 的 TODO B3（「找齊多份文件」要另一條路線）有依據 |
| 信賴區間上界 < 0 | 信賴區間包含 0 | **落差在判官。** jevlike 在這類題目上不如 Jev，改 Taguri 的流程補不回來；要看的是題目寫法或更強的判官 |
| 信賴區間上界 < 0 | 信賴區間上界 < 0 | 兩者都有。分別報各占幾分 |
| 其他 | | 「分不出來」，照實寫 |

### 解讀時要注意（寫進結論）

- 只有 60 題（40 ＋ 20），信賴區間會很寬，「分不出來」是很可能的結果。
- 上游紀錄的 SAG2＋Jev 是上游作者用他們的索引跑的。第 1 步的第 3 點是唯一能確認「我們的索引和他們的一樣」的檢查，不能省。
- 這個題組沒有拿來調過 B0 或 Taguri，結果不要用來調參數。

**結論**：＿＿

## 4. 產出

| 檔案 | 內容 |
|---|---|
| `results/c1-b0-zh.md` | C1 的數字、判定、花費 |
| `results/c2-enterprise-attribution.md` | C2 第 0 步（一定有）與第 1 步（有做才有） |
| `reference/musique-zhtw2/`、`reference/musique-zhcn2/`、`reference/enterprise/` | B0、上游 BM25、SAG2＋Jev 的逐題分數（只有題號和指標） |
| `patches/0002-enterprise-local-scorer.patch` | 只有做第 1 步才有 |
| `results/cost.md` | 加上這兩項的花費 |

## 5. 小結（兩項做完後填）

```
C1 中文：D（繁中）＝ __（信賴區間 __）；B0 繁中 − 英文 __；B0 繁中 − 簡中 __
  判定：Taguri 多掉一塊／兩套都掉（傾向判官／傾向檢索）／分不出來／Taguri 掉得比較少
C2 企業文件，第 0 步：Taguri − SAG2+Jev（60 題，配對）＝ __（信賴區間 __）；整體 470 題 ＝ __
  第 1 步：有做／沒做（原因）
  B0 − SAG2+Jev ＝ __（信賴區間 __）；Taguri − B0 ＝ __（信賴區間 __）
  判定：迴圈的設計／判官／兩者都有／分不出來
花費：$__（帳單核對：是／否）
給 Clarence 的一句話（這兩個結果會不會改變 Taguri 的 TODO 排序）：
要不要把數字回寫到 taguri repo：等 Clarence 決定
```
