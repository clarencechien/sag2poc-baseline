# S0　環境、重現、切分、評測工具

日期：2026-11-10。只用 CPU，花費 $0。

## 門檻（跑之前寫好，見交接文件 S0）

| 門檻 | 結果 | 判定 |
|---|---|---|
| 三個 verify 全部通過 | `verify_paper_results.py`：通過（三個 1,000 題題組、八組對照、470 題 EnterpriseRAG）；`verify_rerank_results.py`：通過（6,000 筆逐題紀錄）；`pytest -q`：93 passed | 過 |
| 自己跑的 BM25，MuSiQue R@5 落在 47.47 ± 0.5 | **47.66** | 過 |
| `evalkit` 重算 SAG2＋Jev，R@5 落在 80.68 ± 0.1、AE@10 落在 69.50 ± 0.2 | R@5 **80.675**、AE@10 **69.50** | 過 |

## 版本

| 東西 | 實際版本 |
|---|---|
| SAG-Benchmark | `Zleap-AI/SAG-Benchmark` @ `bb7f82751ba1576219f9d9ebd021128603d130c3`（和交接文件相同），submodule 在 `vendor/SAG-Benchmark` |
| 多跳題組 | `osunlp/HippoRAG_2` @ `5ec05b38deecc3318bb432c69865959c56058990`；檔案 SHA256 記在 `data/MANIFEST.json` |
| 上游環境 | `uv sync --frozen`，Python 3.13.18，uv 0.11.32 |

題組檔案是直接從 Hugging Face 用釘版網址下載的（網頁版 VM 連得到），再用上游的 `--index-only` 建索引：MuSiQue 11,656 段、2Wiki 6,119 段（6,245 個索引片段）、HotpotQA 9,811 段。

## 自己跑的 BM25（上游程式的第一輪詞彙檢索，不呼叫模型）

`tools/run_upstream_bm25.py`，用 `evalkit` 計分，1,000 題：

| 題組 | R@1 | R@2 | R@5 | R@10 | AE@10 |
|---|---:|---:|---:|---:|---:|
| MuSiQue | 28.71 | 37.50 | 47.66 | 54.50 | 17.50 |
| 2Wiki | 40.43 | 58.18 | 67.10 | 71.63 | 41.60 |

- MuSiQue 和上游 README「Local Reranking」那張表的 BM25（47.66 / 54.50 / 17.50）完全相同，和上游對照紀錄裡的 BM25 階段逐題比對也是 1,000 題零差異。交接文件引用的 47.47 是論文主表的固定參考值，差 0.19 分。
- 2Wiki 和上游同一張表（40.25 / 58.18 / 67.08 / 71.58）差 0.02 到 0.18 分。原因沒有追：上游沒有保留 2Wiki 的 BM25 逐題紀錄可以比對。這個差距不影響後面的配對比較，因為 `reference/` 用的是我們自己跑的這一份。

## `evalkit` 重算上游逐題紀錄

`tools/build_reference.py` 把上游保留的檢索結果對回段落編號，再用 `evalkit` 重新計分。重算結果和上游紀錄裡的逐題指標比對，全部零差異（MuSiQue SAG2＋Jev 8,000 個數值、Jina 兩輪 8,000、Jina 單輪 8,000、qwen3-rerank 8,000、2Wiki SAG2＋Jev 4,000、2Wiki Jina 8,000）。

| 題組 | 方法 | R@5 | R@10 | AE@10 | dev R@5 |
|---|---|---:|---:|---:|---:|
| MuSiQue | BM25 | 47.66 | 54.50 | 17.50 | 44.69 |
| MuSiQue | SAG2＋Jev（兩輪擴展） | 80.67 | 86.62 | 69.50 | 79.94 |
| MuSiQue | SAG2＋Jev，H1 階段 | 77.89 | 83.36 | 63.10 | 77.64 |
| MuSiQue | SAG2＋Jev，H0 階段 | 60.78 | 67.93 | 36.00 | 59.78 |
| MuSiQue | SAG2＋Jina v3.5（兩輪擴展） | 65.61 | 72.12 | 43.80 | 64.17 |
| MuSiQue | BM25＋Jina 單輪 | 63.04 | 67.42 | 34.70 | 62.50 |
| MuSiQue | SAG2＋qwen3-rerank（兩輪擴展） | 55.55 | 71.61 | 44.60 | 54.22 |
| MuSiQue | BM25＋qwen3-rerank 單輪 | 60.17 | 66.88 | 32.50 | 58.64 |
| 2Wiki | BM25 | 67.10 | 71.62 | 41.60 | 66.50 |
| 2Wiki | SAG2＋Jev（兩輪擴展） | 97.35 | 97.42 | 93.10 | 97.75 |
| 2Wiki | SAG2＋Jina v3.5（兩輪擴展） | 80.80 | 86.95 | 67.90 | 83.50 |

要注意的三點：

- **`qwen_three.jsonl` 是 302 平台的 `qwen3-rerank`，不是 Qwen3-Reranker-0.6B**（對照 `summary.json` 確認）。Qwen 0.6B 那組（R@5 62.54）上游只留了總平均，沒有逐題紀錄，所以 `reference/` 裡沒有它。檔名已標成 `sag2_qwen3rerank_302`。
- **Jev 只判一輪（H0）有逐題紀錄**：上游 `controls/sag2_h2` 每題都記了 bm25、h0、h1、h2 四個階段的指標，所以 S2 可以和 Jev H0 做配對比較。這些階段只有指標、沒有檢索名單，`reference/` 直接沿用上游記的數值（H2 階段和最終結果經 `evalkit` 重算一致）。
- MuSiQue dev 300 題比整體難一點：BM25 的 R@5 是 44.69（整體 47.66），SAG2＋Jev 是 79.94（整體 80.67）。

## 切分（之後不改）

| 題組 | 前 N 題的分布 | 整體分布 | 最大差距 | 做法 |
|---|---|---|---:|---|
| MuSiQue（N=300） | 2 跳 161、3 跳 87、4 跳 52 | 518 / 316 / 166 | 2.6 個百分點 | 直接取前 300 題 |
| 2Wiki（N=100） | compositional 45、comparison 25、bridge_comparison 23、inference 7 | 413 / 244 / 235 / 108 | 3.8 個百分點 | 直接取前 100 題 |

都在 5 個百分點以內，所以兩個題組都取檔案前 N 題，上游的 `--offset`、`--limit` 可以照用。`data/musique/splits.json` 有 `dev`（300）、`calib`（前 150）、`check`（後 150）、`dev_zh_source`（前 100）、`test`（700）；`data/2wikimultihopqa/splits.json` 有 `dev`（100）。

## 產出

- `data/`：兩個題組的 `corpus.jsonl`、`questions.jsonl`、`splits.json`，加上 `MANIFEST.json`。段落編號 `pid` 是語料檔的列號（從 0 起算，字串）。金標證據全部對得到語料（0 筆對不到）。
- `evalkit/`：`data.py`、`metrics.py`、`bootstrap.py` 和命令列入口，只依賴 numpy，沒有 import 上游程式碼。
- `reference/`：各方法的逐題分數（只有題號和指標）與 `summary.json`。

## 結論

**繼續**做 S1。三項門檻都通過，資料和評分對得上上游。
