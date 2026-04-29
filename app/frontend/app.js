const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");

function addMessage(role, text, meta = "") {
  const item = document.createElement("div");
  item.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  item.appendChild(bubble);
  if (meta) {
    const tag = document.createElement("div");
    tag.className = "meta";
    tag.textContent = meta;
    item.appendChild(tag);
  }
  chat.appendChild(item);
  chat.scrollTop = chat.scrollHeight;
}

async function sendMessage(message) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, debug: false }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  addMessage("user", text);
  addMessage("assistant", "思考中...");
  const placeholder = chat.lastElementChild;

  try {
    const data = await sendMessage(text);
    placeholder.remove();
    const meta = data.mode === "graph_rag" ? `模式: ${data.mode} | 命中: ${data.row_count ?? 0}` : `模式: ${data.mode}`;
    addMessage("assistant", data.answer || "无输出", meta);
  } catch (err) {
    placeholder.remove();
    addMessage("assistant", `请求失败：${err.message}`);
  }
});

addMessage("assistant", "你好，我是 GraphMuse Copilot。你可以问我模型、论文、数据集或方法总结。");
