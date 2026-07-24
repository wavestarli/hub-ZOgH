(() => {
  const transcript = document.getElementById("transcript");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("question");
  const sendBtn = document.getElementById("sendBtn");
  const sessionPill = document.getElementById("sessionPill");
  const memoryPill = document.getElementById("memoryPill");
  const activePill = document.getElementById("activePill");

  let sessionId = null;
  let closing = false;
  let heartbeatTimer = null;
  let activeTimer = null;

  function addMessage(role, text, extra = null) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    el.textContent = text;
    if (extra && extra.tool_calls && extra.tool_calls.length) {
      const tools = document.createElement("div");
      tools.className = "tools";
      tools.textContent =
        "工具：" +
        extra.tool_calls
          .map((t) => `${t.name}(${JSON.stringify(t.args)})`)
          .join(" · ") +
        (extra.elapsed != null ? ` · ${extra.elapsed.toFixed(1)}s` : "");
      el.appendChild(tools);
    }
    transcript.appendChild(el);
    transcript.scrollTop = transcript.scrollHeight;
  }

  function setMemory(turns, maxTurns) {
    memoryPill.textContent = `记忆 ${turns ?? 0}/${maxTurns ?? 6}`;
  }

  async function refreshActiveCount() {
    try {
      const res = await fetch("/api/sessions");
      if (!res.ok) return;
      const data = await res.json();
      activePill.textContent = `在线 ${data.count}`;
    } catch (_) {
      /* ignore */
    }
  }

  async function createSession() {
    const res = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail || "创建会话失败");
    }
    const data = await res.json();
    sessionId = data.session_id;
    sessionPill.textContent = `会话 ${data.display_name}`;
    sessionPill.title = data.session_id;
    setMemory(data.memory_turns, data.memory_max_turns);
    addMessage(
      "system",
      `会话已建立（${data.session_id}）。关闭本页将销毁短期记忆。`
    );
  }

  async function heartbeat() {
    if (!sessionId || closing) return;
    try {
      const res = await fetch(`/api/session/${sessionId}/heartbeat`, {
        method: "POST",
      });
      if (res.status === 404) {
        sessionPill.textContent = "会话已失效";
        sessionPill.classList.add("warn");
        addMessage("error", "会话已超时销毁，请刷新页面重新建立。");
        stopKeepAlive();
      }
    } catch (_) {
      /* ignore transient network errors */
    }
  }

  function destroySessionBeacon() {
    if (!sessionId || closing) return;
    closing = true;
    const url = `/api/session/${sessionId}/close`;
    const blob = new Blob([JSON.stringify({ reason: "pagehide" })], {
      type: "application/json",
    });
    if (navigator.sendBeacon) {
      navigator.sendBeacon(url, blob);
    } else {
      fetch(url, { method: "POST", body: blob, keepalive: true }).catch(() => {});
    }
  }

  function stopKeepAlive() {
    if (heartbeatTimer) clearInterval(heartbeatTimer);
    if (activeTimer) clearInterval(activeTimer);
    heartbeatTimer = null;
    activeTimer = null;
  }

  function startKeepAlive() {
    heartbeatTimer = setInterval(heartbeat, 15000);
    activeTimer = setInterval(refreshActiveCount, 8000);
    refreshActiveCount();
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!sessionId) {
      addMessage("error", "会话尚未就绪");
      return;
    }
    const question = input.value.trim();
    if (!question) return;

    addMessage("user", question);
    input.value = "";
    sendBtn.disabled = true;
    input.disabled = true;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg =
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail || data) || `请求失败 ${res.status}`;
        addMessage("error", msg);
        if (res.status === 404) {
          sessionPill.textContent = "会话已失效";
          sessionPill.classList.add("warn");
        }
        return;
      }
      addMessage("assistant", data.answer || "(空回答)", data);
      setMemory(data.memory_turns, data.memory_max_turns);
      refreshActiveCount();
    } catch (err) {
      addMessage("error", String(err.message || err));
    } finally {
      sendBtn.disabled = false;
      input.disabled = false;
      input.focus();
    }
  });

  // 关闭 / 刷新 / 离开页面 → 销毁会话
  window.addEventListener("pagehide", destroySessionBeacon);
  window.addEventListener("beforeunload", destroySessionBeacon);

  (async () => {
    try {
      await createSession();
      startKeepAlive();
      input.focus();
    } catch (err) {
      sessionPill.textContent = "连接失败";
      sessionPill.classList.add("warn");
      addMessage("error", String(err.message || err));
    }
  })();
})();
