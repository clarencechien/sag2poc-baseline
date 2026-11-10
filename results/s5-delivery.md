# S5　交付包

日期：2026-11-10。只用 CPU，花費 $0。

交付包已推到 `clarencechien/sag2poc` 的 `main`，一個 commit（`c596fdf`）。

## 內容

| 路徑 | 內容 |
|---|---|
| `evalkit/`、`pyproject.toml` | 指標與資料載入，和本 repo 的 `evalkit/` 相同 |
| `data/` | MuSiQue、2Wiki、MultiHop-RAG 的正規化語料、題目、切分名單與 MANIFEST（共約 16 MB，直接進 git） |
| `reference/` | 各方法的逐題分數、`README.md`（每個檔案是什麼方法）、`summary.json`（各切分的總分） |
| `handoff/A-to-B.md` | S0 到 S4 的數字與結論、Modal 與 SGLang 的實測、jevlike 判斷行為的觀察、`evalkit` 用法 |
| `SAG2POC-handoff.md` | v0.5，已填 S0 到 S5 的結論與 §11.1 |

沒有交付的東西：`adapters/`、`patches/`、`modal_baseline.py`、`tools/`、`vendor/`、`results/`。這些留在 `sag2poc-baseline`。

## 交付前自我檢查

| 檢查 | 結果 |
|---|---|
| `evalkit/` 沒有 import `vendor/` 底下的任何東西 | 過。只用到標準函式庫和 numpy |
| `handoff/A-to-B.md` 沒有 SAG2 的 prompt、公式、參數值、程式碼，也沒有描述 SAG2 內部怎麼運作 | 過。用關鍵字掃過一次，另外人工刪掉兩處會透露上游候選數量與證據數量的描述 |
| `reference/` 只有題號和分數 | 過。13,800 行，欄位只有 `qid` 和八個指標 |
| 乾淨環境 `pip install` 後 `evalkit` 能單獨跑，重算出 SAG2＋Jev 的總分 | 過。新建的虛擬環境、在別的目錄執行：R@5 80.675、AE@10 69.50 |

## 一個要讓 Clarence 知道的判斷

`handoff/A-to-B.md` 寫了「B0 每題約 334 次判斷」和各階段有沒有帶已有證據。前者是交接文件 §3 要求記錄的量測值，後者在交接文件 S3 的校準分析裡已經寫明，所以我保留了。如果覺得這仍然透露太多，可以在開 Session B 之前刪掉 §6.5 和 §6.2 表格上方那一句。

## 結論

**交付完成。** Session A 到此為止，沒有開 Session B，沒有動 `engine/`。
