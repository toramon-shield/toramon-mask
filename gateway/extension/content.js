// Toramon Mask for Web AI — content script（試作 v0.1、ブラウザ上での動作は未検証）
// 方針: 入力欄への「貼り付け」と「Enter 送信」の直前に、端末内の Toramon Gateway (/redact) で個人情報を伏せる。
// 原文はページの外へ出さない。ゲートウェイに届かないときは、貼り付けをそのまま通さず警告だけ出す（fail-closed）。
(() => {
  const DEFAULT_GW = "http://127.0.0.1:8787";
  let gw = DEFAULT_GW;
  try { chrome.storage.sync.get({ gateway: DEFAULT_GW }, (v) => { gw = v.gateway || DEFAULT_GW; }); } catch (e) {}

  async function redact(text) {
    const r = await fetch(gw + "/redact", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    if (!r.ok) throw new Error("gateway " + r.status);
    const d = await r.json();
    return { text: d.redacted_text, counts: d.counts || {} };
  }

  function toast(msg, warn) {
    const el = document.createElement("div");
    el.textContent = msg;
    el.style.cssText = "position:fixed;right:16px;bottom:16px;z-index:2147483647;background:" + (warn ? "#8a2b1f" : "#1f5f4a") + ";color:#fff;padding:10px 14px;border-radius:6px;font:13px/1.5 sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.3)";
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3500);
  }

  function summarize(counts) {
    const LB = { private_person: "氏名", private_address: "住所", private_email: "メール", private_phone: "電話", private_date: "日付", account_number: "識別番号", private_url: "URL", secret: "秘密情報" };
    return Object.entries(counts).map(([k, v]) => (LB[k] || k) + " " + v).join("、");
  }

  function isEditable(el) {
    return el && (el.tagName === "TEXTAREA" || el.isContentEditable);
  }

  function readText(el) { return el.tagName === "TEXTAREA" ? el.value : el.innerText; }

  function writeText(el, text) {
    if (el.tagName === "TEXTAREA") {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set;
      setter.call(el, text);
      el.dispatchEvent(new Event("input", { bubbles: true }));
    } else {
      el.focus();
      const sel = window.getSelection(); sel.selectAllChildren(el);
      document.execCommand("insertText", false, text);
    }
  }

  // 1) 貼り付け: クリップボードの文章を伏せてから挿入する
  document.addEventListener("paste", async (ev) => {
    const el = ev.target && (isEditable(ev.target) ? ev.target : ev.target.closest && ev.target.closest("textarea,[contenteditable='true']"));
    if (!el) return;
    const raw = (ev.clipboardData || window.clipboardData).getData("text/plain");
    if (!raw || raw.length < 2) return;
    ev.preventDefault(); ev.stopImmediatePropagation();
    try {
      const { text, counts } = await redact(raw);
      document.execCommand("insertText", false, text);
      const n = Object.values(counts).reduce((a, b) => a + b, 0);
      if (n) toast("Toramon: 貼り付け前に " + n + " 箇所を伏せました（" + summarize(counts) + "）");
    } catch (e) {
      toast("Toramon Gateway に接続できないため貼り付けを止めました（" + e.message + "）", true);
    }
  }, true);

  // 2) 送信（Enter）: 入力欄の全文を伏せてから本来の送信を続ける
  let bypass = false;
  document.addEventListener("keydown", async (ev) => {
    if (bypass || ev.key !== "Enter" || ev.shiftKey || ev.isComposing) return;
    const el = isEditable(ev.target) ? ev.target : null;
    if (!el) return;
    const raw = readText(el);
    if (!raw || raw.length < 2) return;
    ev.preventDefault(); ev.stopImmediatePropagation();
    try {
      const { text, counts } = await redact(raw);
      if (text !== raw) {
        writeText(el, text);
        const n = Object.values(counts).reduce((a, b) => a + b, 0);
        toast("Toramon: 送信前に " + n + " 箇所を伏せました（" + summarize(counts) + "）");
      }
      bypass = true;
      el.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", code: "Enter", bubbles: true, cancelable: true }));
      bypass = false;
    } catch (e) {
      toast("Toramon Gateway に接続できないため送信を止めました（" + e.message + "）", true);
    }
  }, true);
})();
